# MoonEP Dispatch V1 Migration Design

## 1. Status

This document defines the approved design for migrating the tested MoonEP dispatch
scatter implementation into the `TileXR-moonep-migration` main framework.

Target repository baseline:

```text
repository: TileXR-moonep-migration
branch:     main
commit:     e9b33c1
CANN:       9.1.0
runtime:    NPU driver 25.5.0 or later
hardware:   Ascend950 PR and Ascend950 DT
```

Reference implementation:

```text
reference/TileXR-moonep-dispatch/TileXR-moonep-dispatch
```

The local reference implementation does not contain Git metadata. Before source
migration starts, generate and preserve a SHA256 manifest for the files used by
the migration. The manifest identifies the source tree associated with the
existing 128-rank correctness and performance evidence.

The following path is explicitly excluded from analysis and migration:

```text
reference/TileXR
```

The design has been approved incrementally. Implementation must not silently
change the contracts in this document. Any required material change returns to
design review before code changes continue.

## 2. Goal

Replace the `TileXRMoonEpDispatchV1` local stub in the main MoonEP Runtime with a
native Ascend C UDMA scatter implementation while preserving the existing main
framework:

- keep the public `DispatchV1` entry point instead of adding `DispatchV2`;
- keep FP16 and BF16 hidden payload support;
- dispatch FP32 route weights through a second `DispatchV1` call;
- consume the current Planner `dst` encoding;
- expose the implementation through the existing C ABI and Torch Runtime;
- preserve the tested reference transport and Kernel performance structure;
- validate up to 128 ranks first while keeping the scheduling design ready for
  a later 512-rank runtime expansion.

## 3. Scope

This phase makes these stages native:

```text
Planning
Dispatch
```

These stages remain stubs:

```text
PrefetchWeight
Combine
ReduceGrad
```

Consequently, this phase may claim native Planner and Dispatch correctness. It
must not claim that the full MoonEP forward/backward pipeline is native.

The following features are out of scope:

- a new `DispatchV2` API;
- one-Kernel hidden and route-weight transfer;
- original MoonEP negative-`dst` weight-only behavior;
- original MoonEP dispatch epilogue and duplicate hidden expansion;
- zero filling destination holes;
- quantization;
- tensor parallel dispatch;
- shared experts;
- unordered multi-stream execution on one communicator/workspace;
- a claim of 512-rank hardware validation.

The original MoonEP negative-`dst` behavior remains a documented legacy design
point. This migration sends every route after decoding its signed `dst` value.

## 4. Invocation And Ownership

The authoritative invocation path is:

```text
Torch API
  -> tilexr_moonep.runtime ctypes binding
  -> TileXRMoonEpDispatchV1 Host validation and launch
  -> embedded Ascend C Kernel binary
  -> registered UDMA workspace
```

The direct-launch Host layer owns validation, layout construction, magic
allocation, Kernel argument construction, binary registration, and asynchronous
stream launch.

The Torch context owns:

- communicator;
- raw device allocation for Dispatch workspace;
- aligned Dispatch workspace pointer and byte capacity;
- UDMA registration handle;
- the single-stream ownership state;
- pending tensor, Plan, and raw workspace references;
- failed/poisoned state after a fatal asynchronous error.

The Planner workspace and Dispatch registered workspace are independent
allocations and must not be reused for each other.

Cleanup order is:

```text
quiesce all work
  -> inspect Planner/Dispatch status
  -> TileXRUDMAUnregister
  -> release the raw workspace owner
  -> destroy communicator
```

After a completion timeout, the communicator/workspace is poisoned and is not
eligible for normal reuse. The distributed launcher terminates the job rather
than attempting continued execution with possible late remote writes.

## 5. Public V1 ABI

`TileXRMoonEpDispatchArgsV1` is extended at its tail:

```cpp
typedef struct TileXRMoonEpDispatchArgsV1 {
    uint32_t structSize;
    uint32_t abiVersion;
    TileXRCommPtr comm;
    const TileXRMoonEpPlanV1 *plan;
    const TileXRMoonEpTensorV1 *input;
    TileXRMoonEpTensorV1 *output;
    uint64_t flags;
    void *registeredWorkspace;
    uint64_t registeredWorkspaceBytes;
} TileXRMoonEpDispatchArgsV1;
```

The ABI version and library SONAME remain V1. The function signature remains
unchanged. Host code must check `structSize` before reading appended fields. A
structure large enough for the legacy V1 fields but too short for the appended
workspace fields returns `TILEXR_MOONEP_ERROR_NOT_SUPPORTED`. A structure too
short even for the legacy fields returns `TILEXR_MOONEP_ERROR_INVALID_ARGUMENT`.
Neither case may read past the declared size.

The workspace query is:

```cpp
int TileXRMoonEpDispatchGetWorkspaceSizeV1(
    TileXRCommPtr comm,
    int64_t s,
    int64_t k,
    int64_t h,
    uint32_t hiddenDtype,
    uint64_t *workspaceBytes,
    uint64_t *workspaceAlignment);
```

The query accepts FP16 or BF16 as `hiddenDtype`, performs checked arithmetic,
and returns the maximum space needed by hidden and weight Dispatch modes.
`workspaceAlignment` is 2 MiB.

After migration, capabilities are:

```text
nativeStages = PLANNING | DISPATCH
stubStages   = PREFETCH_WEIGHT | COMBINE | REDUCE_GRAD
```

## 6. Tensor Contracts

### 6.1 Hidden Mode

```text
input dtype:       FP16 or BF16
input shape:       [S, H]
output dtype:      same as input
output shape:      [NvS, H]
NvS:               Plan.dispatchedCapacity == S*K
source index:      routeId / K
bytes per record:  H * 2
```

### 6.2 Route-Weight Mode

```text
input dtype:       FP32
input shape:       [S, K]
output dtype:      FP32
output shape:      [NvS]
NvS:               Plan.dispatchedCapacity == S*K
source index:      routeId
bytes per record:  4
```

The Host selects one of these modes only after an exact dtype and shape match.
No other combination is accepted. The internal Kernel ABI receives an explicit
payload mode and row byte count; the Kernel must not infer weight mode from a
fabricated hidden dimension.

All three supported payload dtypes are transported bit-for-bit. Dispatch does
not perform arithmetic or dtype conversion.

Torch tensors must be dense contiguous device tensors on the communicator's
current NPU, with storage offset zero. Public C callers own the equivalent
contiguity and device-memory guarantee. Input, output, Plan, and Dispatch
workspace ranges must not overlap.

## 7. Plan And `dst` Contract

Dispatch consumes only the Plan fields needed for identity and routing. `cu`,
`expertsToCopy`, and `remoteStats` remain available to other stages but are not
used for Dispatch data movement.

For every route:

```cpp
int64_t raw = encoded >= 0 ?
    static_cast<int64_t>(encoded) :
    -static_cast<int64_t>(encoded) - 1;

targetRank = raw / NvS;
targetSlot = raw % NvS;
```

The legal range is:

```text
0 <= raw < world*NvS
```

All `S*K` routes are sent, including routes whose encoded value is negative.
Negative encoding remains metadata about duplicate expert routing; it does not
suppress the hidden payload in this migration.

Across all ranks, decoded destinations must form a one-writer mapping for every
physical `(targetRank, targetSlot)`. The Planner and its reference tests own this
global invariant. Dispatch validates every decoded route before address
calculation but does not perform a global Host-side bijection check.

An invalid route is skipped, recorded in DFX, and makes the invocation output
invalid. The Kernel still emits completion notifications to avoid leaving other
ranks in an infinite wait. No zero fill is performed for missing destination
slots.

## 8. Planner-To-Dispatch Error Propagation

Planning and Dispatch execute on the same stream, so Dispatch observes local
Plan writes without a Host synchronization.

`plan.status` is a sticky device status for the Plan lifecycle:

- Planning initializes and writes Planner status;
- Dispatch never clears an existing nonzero status;
- Dispatch atomically records its first stage-specific error only when status is
  still success;
- the Torch Runtime reads the status only after quiescing the stream.

If Dispatch observes an upstream Planner failure, it does not issue payload
PUTs. It still participates in the completion protocol, records an upstream
error in DFX, and returns an invalid output. This prevents one failed rank from
causing otherwise healthy ranks to wait forever.

Host validation failures occur before launch. Therefore all ranks must supply
the same valid shapes, mode, workspace capacity, and call sequence. A launcher
or higher runtime must propagate pre-launch failures across ranks; otherwise
peers can wait until the one-minute device timeout.

## 9. Workspace Layout

The Runtime allocates one workspace large enough for both payload modes:

```text
hidden source bytes       = S * H * 2
hidden scratch slot bytes = NvS * H * 2
weight source bytes       = NvS * 4
weight scratch slot bytes = NvS * 4
scratch buffers           = 2
```

Each mode has an active source/scratch layout. The common tail begins after the
maximum active data extent and contains:

```text
completionFlags[512]
signalSource[64][64B]
hiddenProfile[64]
weightProfile[64]
hiddenDfx[64]
weightDfx[64]
shared kernel status
```

Internal regions are 64-byte aligned. The returned registration region is
rounded to 2 MiB. Reserving 512 completion flags now keeps the workspace layout
ready for the later rank-limit expansion without changing current communicator
support.

The Runtime allocates `workspaceBytes + alignment - 1`, derives an aligned view,
retains the raw allocation owner, zeroes the aligned region once, and registers
the aligned pointer. There is no per-round allocation, registration, memset, or
layout query.

Each `DispatchV1` invocation validates or reconstructs its small active layout
on the Host. This is checked integer arithmetic only and does not insert a
device operation or stream synchronization between Kernel launches.

For multi-rank execution, the UDMA registry must contain the same workspace base
and at least the required byte capacity for every rank. Single-rank execution
uses the same layout but skips UDMA registration.

## 10. QP-B Transport Contract

Multi-rank MoonEP Dispatch requires:

```bash
TILEXR_UDMA_ROUTE_POLICY=moonep_qpb
```

The environment variable must be set before communicator initialization.
`moonep_qpb` is a communicator construction policy, not a Dispatch argument.

The transport contract is:

- require a valid `localRankSize`;
- require HCCL root information and topology data;
- same-node peers select the local EID through the topology edge and port map;
- remote-node peers select an EID with exactly six ports;
- ranks exchange local route choices to derive the matching remote EID;
- create one peer-specific QP/CQ path per remote rank;
- expose one selected QP per peer to the Kernel (`qpNum == 1`);
- fail QP-B initialization when required topology is unavailable instead of
  falling back to an arbitrary EID.

`ExtraFlag::UDMA_MOONEP_QPB` is a runtime-published capability bit. Callers must
not set it manually. It is set only after successful QP-B route and queue
construction. Multi-rank Dispatch requires both `UDMA` and
`UDMA_MOONEP_QPB`; single-rank Dispatch does not.

On Ascend950 PR and DT, QP-B registered workspaces use `nonPin=1`. Other chips
and non-QP-B UDMA registrations retain `nonPin=0`.

## 11. Required UDMA Device Operations

The migration includes the transport capabilities required by the reference
Kernel without replacing unrelated main transport behavior:

- deferred PUT on the selected QP;
- explicit SQE completion flag;
- strong/ordered completion for the final flag PUT;
- explicit QP doorbell flush;
- quiet status reporting;
- SQ segmentation and reclaim before unconsumed entries can be overwritten.

Payload PUTs use normal completion. The final per-peer completion-flag PUT uses
ordered completion and shares the last doorbell with the final payload segment.
Even a peer receiving no payload records receives a completion flag for the
current magic.

The existing GET, signal, non-deferred PUT, and ordinary UDMA behavior must not
regress.

## 12. Kernel Scheduling

The Kernel uses:

```text
AIV block count: 64
rank group size: 64
group count:     ceil(rankSize / 64)
maximum design group count: 8
```

For each phase:

```text
currentGroup = rank / 64
targetGroup  = (currentGroup + phase) % groupCount
peer         = targetGroup * 64 + aivId
```

An AIV is idle in a phase when `peer >= rankSize`. The current main runtime
continues to reject more than 128 ranks, so hardware execution initially uses at
most two phases. The Kernel and schedule tests retain all eight phases for the
future 512-rank target.

`rankGroupSize` and `aivCoreNum` are internal constants, not public Dispatch V1
arguments. Host launch rejects devices with fewer than 64 vector cores.

## 13. Kernel Flow And Synchronization

The Kernel flow is:

1. validate uniform scalar parameters and device transport metadata;
2. stage the current payload into registered source memory;
3. execute `AscendC::SyncAll<true>()` so every sender sees complete staging;
4. initialize this AIV's signal source for the current magic;
5. select vector or scalar-tiled route processing;
6. issue local copies and remote deferred payload PUTs;
7. emit one ordered completion flag to every remote peer;
8. wait for assigned source-peer completion flags;
9. execute `AscendC::SyncAll<true>()` so output copying starts only after every
   source group is complete;
10. copy the current scratch slot into the public output;
11. quiet/reclaim issued queues and record errors;
12. write DFX and optional Profile records.

Both `SyncAll` operations from the reference implementation are retained in the
initial migration. Removing either barrier is a separate optimization requiring
a new ownership proof and target-hardware evidence.

Every GM/UB copy supports bounded tails. FP32 weights are four-byte records and
must not rely on hidden-row alignment. No scalar GM `GetValue/SetValue` loop may
replace the bounded data-movement path for payload bytes.

## 14. Route Selection Paths

The reference vector route-selection fast path is retained when:

- `NvS` is a power of two;
- `NvS` satisfies vector index limits;
- route metadata fits the UB budget;
- the relevant Ascend C vector API granularity is legal.

All other legal shapes use the scalar-tiled fallback. Both paths perform the
same complete signed-`dst` decode and target validation before issuing a copy or
PUT.

The benchmark shape `S=128, K=16, NvS=2048` must select the vector path. Tests
also include non-power-of-two `NvS` and verify identical output semantics.

## 15. Magic, Stream, And Concurrency

Every `DispatchV1` Host launch obtains a new magic internally through
`TileXRCommNextMagic`. Magic is not part of the public ABI.

Hidden and route weights are two collective Dispatch calls with two distinct
magic values:

```text
DispatchV1(hidden)
DispatchV1(routeWeights)
```

All ranks must execute the same call count and mode order. Route-weight presence
is collective: it cannot be present on only a subset of ranks.

The workspace selects scratch with `magic % 2`. Multiple calls may be in flight
only as ordered work on one stream. The same communicator/workspace cannot be
used concurrently from different streams or from two Buffer objects. Stream
ownership is enforced by the context/workspace owner rather than only by an
individual Buffer.

Changing streams requires a completed quiesce boundary. Outputs may be consumed
immediately by later work on the same stream or after an explicit event/sync on
another stream.

## 16. One-Minute Completion Timeout

The completion-wait phase has one total deadline of 60 seconds. The deadline
starts when an AIV enters completion waiting and is shared across every peer
wait performed by that AIV. It is not 60 seconds per peer.

The public V1 ABI does not expose a timeout argument. Host launch passes an
internal `completionTimeoutTicks` Kernel argument. Host conversion from 60
seconds to counter ticks must use CANN 9.1/Ascend950 version-matched timer
evidence and checked arithmetic. It must not use a guessed device frequency.

The implementation must verify the exact `AscendC::GetSystemCycle()` unit or use
another documented device time source before the conversion is finalized. A
shorter test-only timeout may be compiled for fault tests; production remains
fixed at 60 seconds.

On timeout:

- set `kMoonEpDispatchDfxCompletionTimeout`;
- record peer, phase, expected magic, and last observed completion value;
- mark the Plan/Dispatch status as failed;
- treat the output as invalid;
- let every AIV converge through the retained barrier and finish the Kernel;
- poison the Runtime context after status inspection;
- reject subsequent Dispatch calls and terminate the distributed job.

This replaces the reference Kernel's unbounded polling loop.

## 17. DFX And Profiling

DFX is correctness and communication diagnostics. It is always enabled and is
not a user tensor output. Profile records are optional performance diagnostics.

Each payload mode owns 64 DFX records and, when profiling is enabled, 64 Profile
records. Hidden and weight calls therefore do not overwrite one another's last
diagnostic record.

DFX records include:

- marker, version, record size, payload mode, rank, and AIV;
- DFX flags;
- first invalid route and raw destination;
- first quiet error and phase;
- completion-timeout peer, phase, expected magic, and observed flag;
- expected and processed route counts;
- current magic.

Every normally entered Kernel writes a complete DFX record. A successful round
writes zero flags, so stale failures cannot be interpreted as current failures.
Changing the internal record layout increments the DFX record version; it does
not change the public MoonEP V1 ABI.

Profile records preserve the reference counters and eight timeline offsets. A
stage duration is calculated from adjacent time points, not interpreted as an
independent cumulative duration.

Host API return values cover synchronous validation, binary registration, and
launch errors only. Planner/Dispatch asynchronous errors are surfaced after
stream completion through sticky Plan status, with DFX providing detailed
diagnosis.

Official latency and diagnostic profiling use separate otherwise-identical
runs:

```text
profiling OFF -> acceptance latency
profiling ON  -> last-round timeline and DFX validation
```

There is no D2H diagnostic copy inside the hot loop.

## 18. Torch Runtime Behavior

Torch retains the existing public flow:

```text
planning
dispatch hidden
dispatch route weights when provided
prefetch weight stub
expert callback
apply dispatched route weights when requested
combine stub
```

The second Dispatch call reuses the same Plan, communicator, stream, and
registered workspace. Each call selects its own payload mode and active layout.

The Runtime must:

- extend the ctypes V1 struct exactly like the C header;
- query, over-allocate, align, zero, and register Dispatch workspace once;
- retain the raw allocation separately from the aligned view;
- enforce context-level single-stream ownership;
- retain inputs, outputs, Plan, and workspace through completion;
- check sticky Plan/Dispatch status after quiesce;
- preserve detailed DFX until it has been inspected;
- reject operations after a fatal completion timeout;
- unregister before releasing the raw workspace.

No per-round workspace query, allocation, registration, memset, or synchronize
is permitted.

## 19. Build And Packaging

New implementation code belongs under:

```text
src/moonep/dispatch/common
src/moonep/dispatch/host
src/moonep/dispatch/kernels
src/moonep/dispatch/cmake
```

It is built into `libtilexr-moonep.so`. The MoonEP Runtime must not require the
whole `libtilexr-ep.so` merely to obtain Dispatch.

The build reuses the reference direct-launch binary embedding approach and
keeps a `TILEXR_MOONEP_DISPATCH_ENABLE_PROFILING` option. The normal
`TILEXR_BUILD_MOONEP=ON` path builds Planner and Dispatch. `TILEXR_BUILD_EP`
remains independent.

The migration copies and adapts selected source into the active tree. Active
targets must not include or link files from `reference/`.

## 20. Validation

### 20.1 Host And Static Tests

- C header compatibility and ctypes layout;
- legacy short V1 struct rejection without out-of-bounds access;
- workspace checked arithmetic, alignment, and both payload layouts;
- hidden/weight shape and dtype validation;
- positive and negative `dst` decode;
- schedule coverage for 1, 8, 16, 64, 65, 128, 256, and 512 ranks;
- vector eligibility and scalar fallback reasons;
- QP-B same-node and remote-node EID selection;
- PR/DT `nonPin=1` and other-path `nonPin=0`;
- missing QP-B capability rejection;
- DFX record parsing and stale-record rejection;
- one-minute timeout logic with a short test-only deadline;
- capability transition from Dispatch stub to native.

### 20.2 Kernel And Hardware Tests

- target CANN 9.1/Ascend950 Kernel compile;
- FP16 and BF16 hidden bit-exact output;
- FP32 route-weight bit-exact output;
- all-local, all-remote, balanced, biased, and duplicate routing;
- negative `dst` routes send both hidden and weights;
- power-of-two vector and non-power-of-two scalar-tiled paths;
- single-rank local mode;
- 8-rank and 16-rank functional tests;
- 128-rank reference-shape correctness and performance;
- repeated magic and alternating scratch over multiple rounds;
- missing completion fault injection and DFX timeout;
- 950PR and 950DT UDMA registration compatibility.

Hardware, simulator, compile, and static evidence are reported separately.
Static 512-rank schedule tests are not 512-rank hardware proof.

## 21. Performance Acceptance

The fixed benchmark is:

```text
BS:       128
H:        3584
K:        16
warmup:   10
measured: 100
mode:     hot loop
```

Workspace preparation, Plan generation, registration, barriers outside the
operator, and D2H diagnostics occur outside the measured loop.

For each measured round, collect every rank's Kernel E2E and define the round
latency as the maximum rank latency. Across the 100 round maxima report:

```text
mean
minimum
p50
p95
maximum
```

Also report Host E2E average/minimum and Kernel E2E average/minimum. Report
hidden-only, weight-only, and hidden-plus-weight pair performance separately.

The profiling-OFF 128-rank BF16 hidden result is compared with the source
manifest's result on the same server model, CANN/driver environment, shape, and
measurement method. Mean and p50 regression must each be no more than 5%.
FP16 and weight-only paths have no source baseline and are reported without a
relative threshold in this phase.

The profiling-ON run preserves and parses only the final round for each payload
mode. Timeline charts use time-point differences.

## 22. 512-Rank Target

The final product target is 512-rank hardware execution, but this phase retains
the main runtime's 128-rank limit.

The initial migration must not introduce new hard-coded two-phase assumptions.
It preserves eight scheduling phases, 512 completion slots, checked address
encoding, and static 256/512 tests.

The later 512-rank phase must address the complete runtime rather than only the
Dispatch Kernel:

- separate legacy AllToAll matrix capacity from global rank capacity;
- expand `CommArgs`, peer arrays, magic arrays, and UDMA registry safely;
- validate ABI and device memory growth;
- validate 511 peer-specific queues per rank;
- validate socket exchange and communicator startup stability;
- validate workspace registration across all EIDs;
- validate eight-phase completion and tail scheduling;
- run correctness, stability, and performance on 512-rank hardware.

Until that work succeeds, reports use these exact claims:

```text
128P: hardware validated
512P: design target, not yet hardware validated
```

## 23. Documentation Requirements

User and test documentation must state:

- `TILEXR_UDMA_ROUTE_POLICY=moonep_qpb` is set before communicator init;
- all ranks use the same payload mode and call sequence;
- one communicator/workspace is bound to one ordered stream;
- Dispatch workspace alignment, registration, and cleanup ownership;
- Host success is asynchronous launch success, not device completion success;
- DFX is read only after quiesce;
- completion timeout is one minute for the whole wait phase;
- profiling changes the measured Kernel and uses a separate run;
- original MoonEP negative weight-only behavior is not implemented;
- only Planning and Dispatch are native in this phase;
- 512P remains a target until hardware validation completes.
