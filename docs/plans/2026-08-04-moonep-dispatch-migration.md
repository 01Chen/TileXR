# MoonEP Dispatch V1 Migration Implementation Plan

## Goal

Implement the approved direct-launch MoonEP Dispatch V1 design in
`TileXR-moonep-migration` without linking active targets to `reference/` and
without changing the current 128-rank runtime limit.

Authoritative design:

```text
docs/specs/2026-08-04-moonep-dispatch-migration-design.md
docs/specs/2026-08-04-moonep-dispatch-migration-design-zh.md
```

Development branch:

```text
codex/moonep-dispatch-v1-migration
```

## Scope And Non-Goals

The implementation makes Planning and Dispatch native. It preserves
PrefetchWeight, Combine, and ReduceGrad as stubs. It supports FP16/BF16 hidden
payloads and FP32 route weights through two ordered Dispatch V1 calls.

This plan does not add Dispatch V2, remove either Kernel `SyncAll`, implement
the original MoonEP negative-`dst` weight-only behavior, change the main rank
limit, claim 512-rank hardware support, or run tests on a remote server before
the user approves the local implementation.

## Task 1: Pin The Reference Inputs

### Objective And Role

Create a reproducible record of the non-Git reference tree used for migration.

### Background And Prerequisites

The reference Dispatch checkout has no `.git` directory. The approved spec
requires source hashes before copying implementation files.

### Modification Scope

- `docs/reference-manifests/`
- selected files under `reference/TileXR-moonep-dispatch/TileXR-moonep-dispatch`
  are read-only inputs

### Constraints And Non-Goals

- Do not read or hash `reference/TileXR`.
- Do not include or link active code from any reference path.
- Do not modify the reference tree.

### Acceptance And Verification

- Manifest contains relative path, SHA256, and byte size for every copied or
  behaviorally referenced source file.
- A verification command recomputes the hashes with no mismatch.

### Artifacts And Interfaces

- `docs/reference-manifests/2026-08-04-tilexr-moonep-dispatch.sha256`

## Task 2: Add QP-B Transport Capabilities

### Objective And Role

Provide the communicator and device-side UDMA capabilities required by the
Dispatch Kernel while preserving all existing main transport behavior.

### Background And Prerequisites

The source transport adds MoonEP QP-B routing, per-peer queues, deferred PUT,
ordered completion, explicit doorbell flush, quiet status, route selection by
topology/port count, and PR/DT non-pin registration. Main has newer IPC/SDID and
registration behavior that remains authoritative.

### Modification Scope

- `src/include/comm_args.h`
- `src/include/tilexr_udma.h`
- `src/include/tilexr_udma_types.h` only when required by the selected APIs
- `src/comm/tilexr_comm.cpp`
- `src/comm/udma/tilexr_udma_transport.{h,cpp}`
- `src/comm/udma/tilexr_udma_layout.{h,cpp}`
- focused tests under `tests/comm/unit` and `tests/udma/unit`

### Constraints And Non-Goals

- Keep `TILEXR_MAX_RANK_SIZE == 128`.
- Preserve main IPC PID/SDID selection, UDMA best-effort initialization,
  registration rollback, and existing API semantics.
- Do not replace complete main transport files with reference copies.
- Ordinary UDMA routes must remain usable by existing callers.

### Acceptance And Verification

- `UDMA_MOONEP_QPB` is set only after successful QP-B initialization.
- Same-node topology route and remote six-port EID selection have unit tests.
- QP-B creates peer-specific QP/CQ state and publishes `qpNum == 1` per peer.
- Deferred PUT, ordered flags, doorbell flush, and quiet-status helpers compile
  in source/host fixtures.
- 950PR/950DT QP-B registration selects `nonPin=1`; all other paths remain zero.
- Existing UDMA registry/layout tests pass.

### Artifacts And Interfaces

- `ExtraFlag::UDMA_MOONEP_QPB`
- `TileXRUDMATransport::IsMoonEpQpBMode()`
- device-side deferred/ordered UDMA helper APIs

## Task 3: Define Dispatch Common Contracts

### Objective And Role

Implement the shared Host/Kernel layout, scheduling, mode, DFX, and Profile
contracts before writing the launcher and Kernel.

### Background And Prerequisites

Depends on the approved ABI and workspace design, but can be host-unit tested
before target Kernel compilation.

### Modification Scope

- `src/moonep/dispatch/common/dispatch_common.h`
- `src/moonep/dispatch/common/dispatch_schedule.h`
- `src/moonep/dispatch/common/dispatch_profile.h`
- `src/moonep/dispatch/host/dispatch_layout.{h,cpp}`
- new focused unit tests under `tests/moonep/unit`

### Constraints And Non-Goals

- Use checked 64-bit arithmetic and explicit byte units.
- Reserve 512 completion flags while Host runtime remains limited to 128 ranks.
- Keep hidden and weight DFX/Profile records separate.
- Keep internal structures C++14-compatible on Host.

### Acceptance And Verification

- Hidden and weight layouts match the approved formulas and 2 MiB alignment.
- Overflow, zero, invalid dtype, invalid capacity, and undersized workspace fail.
- Schedule tests cover 1, 8, 16, 64, 65, 128, 256, and 512 ranks.
- Positive/negative destination decoding has boundary tests.
- DFX/Profile sizes, versions, and timeline indices have static assertions.

### Artifacts And Interfaces

- `DispatchPayloadMode`
- `MoonEpDispatchLayout`
- schedule helpers used identically by Host tests and Kernel

## Task 4: Extend The Public V1 Host ABI

### Objective And Role

Expose workspace requirements and the native Dispatch inputs without creating a
new API version.

### Background And Prerequisites

Depends on Task 3 layout definitions.

### Modification Scope

- `src/include/tilexr_moonep.h`
- `src/moonep/host/tilexr_moonep.cpp`
- `tests/moonep/unit/test_tilexr_moonep_{c_header,abi_layout,host,sources}.cpp`

### Constraints And Non-Goals

- Append fields to Dispatch V1 only.
- Check `structSize` before reading appended fields.
- Preserve V1 ABI version and SONAME.
- Keep non-Dispatch stage stub behavior unchanged.

### Acceptance And Verification

- Workspace query returns maximum hidden/weight space and 2 MiB alignment.
- Legacy-size structs return the exact approved status without over-read.
- Host accepts only the two exact payload contracts.
- Capabilities report Planning and Dispatch as native only.
- Rank/Plan/comm, dtype/shape, core-count, workspace, registry, and QP-B errors
  map to documented MoonEP statuses.

### Artifacts And Interfaces

- extended `TileXRMoonEpDispatchArgsV1`
- `TileXRMoonEpDispatchGetWorkspaceSizeV1`

## Task 5: Implement The Generic Dispatch Kernel

### Objective And Role

Adapt the reference BF16 hidden scatter Kernel into one bitwise-copy Kernel that
supports hidden and route-weight modes and the approved error behavior.

### Background And Prerequisites

Depends on Tasks 2 and 3. The reference Kernel is the algorithmic baseline.

### Modification Scope

- `src/moonep/dispatch/kernels/tilexr_moonep_dispatch_kernel.cpp`
- shared common headers from Task 3
- Kernel-facing UDMA headers from Task 2

### Constraints And Non-Goals

- Retain both `AscendC::SyncAll<true>()` calls.
- Retain vector slot selection and scalar-tiled fallback.
- Decode negative `dst` and send every route.
- Transport bytes without numeric conversion.
- Support four-byte weight tails through legal bounded data movement.
- Do not insert a per-round clear or device-wide synchronization.

### Acceptance And Verification

- Internal ABI order matches launcher static assertions.
- Hidden source index is `routeId/K`; weight source index is `routeId`.
- Every remote peer receives an ordered completion flag even with zero payload.
- Completion wait uses one 60-second total deadline, with a test-only shorter
  build constant.
- Timeout, invalid route, route-count mismatch, upstream Planner error, and
  quiet error set sticky status and detailed DFX.
- Output is copied only after the retained global completion barrier.
- Profiling-off keeps layout ABI stable.

### Artifacts And Interfaces

- `tilexr_moonep_dispatch_kernel`
- exact internal launch argument structure

## Task 6: Add Embedded Direct Launch And Build Wiring

### Objective And Role

Build, embed, register, and asynchronously launch the new Kernel from
`libtilexr-moonep.so`.

### Background And Prerequisites

Depends on Tasks 3 and 5. Reuse the reference EP binary-registration mechanism
but keep ownership in the MoonEP module.

### Modification Scope

- `src/moonep/dispatch/host/dispatch_launch.{h,cpp}`
- `src/moonep/dispatch/cmake/embed_dispatch_kernel.cmake`
- `src/moonep/CMakeLists.txt`
- generated embed source only in the build tree

### Constraints And Non-Goals

- `TILEXR_BUILD_MOONEP=ON` builds Planner and Dispatch.
- `TILEXR_BUILD_EP` remains independent.
- Do not add `reference/` include/link paths.
- Do not add toolkit `devlib` to runtime RPATH.

### Acceptance And Verification

- Target compiler builds the selected CANN 9.1/Ascend950 Kernel.
- Embedded binary symbol, registered Kernel name, and internal ABI agree.
- Host obtains one new magic for every launch and performs no stream sync.
- Profiling option produces distinct profiling-on/off Kernel builds.

### Artifacts And Interfaces

- embedded Kernel image inside `libtilexr-moonep.so`
- direct-launch Host helper used by `TileXRMoonEpDispatchV1`

## Task 7: Complete Native Dispatch Host Integration

### Objective And Role

Replace the Dispatch stub with full validation and native launch while
preserving the other stage stubs.

### Background And Prerequisites

Depends on Tasks 2, 3, 4, and 6.

### Modification Scope

- `src/moonep/host/tilexr_moonep.cpp`
- Dispatch Host helpers under `src/moonep/dispatch/host`
- focused Host tests and fakes

### Constraints And Non-Goals

- Host remains asynchronous.
- Single rank skips UDMA/QP-B registration checks but still requires workspace.
- Multi-rank requires registered workspace and QP-B capability.
- Do not synchronize to inspect device Plan contents before launch.

### Acceptance And Verification

- Dispatch stub memcpy/memset path is no longer reachable.
- Hidden and weight calls construct the correct internal mode and layout.
- Registry contains every peer workspace region for the required byte span.
- Plan status is sticky across Planning, hidden Dispatch, and weight Dispatch.
- Existing PrefetchWeight, Combine, and ReduceGrad tests retain stub behavior.

### Artifacts And Interfaces

- native `TileXRMoonEpDispatchV1`

## Task 8: Integrate Torch Workspace And Lifecycle

### Objective And Role

Make the Torch context allocate, align, register, retain, and clean up the
Dispatch workspace and enforce collective stream semantics.

### Background And Prerequisites

Depends on Tasks 4 and 7.

### Modification Scope

- `integrations/moonep_torch/tilexr_moonep/abi.py`
- `integrations/moonep_torch/tilexr_moonep/runtime.py`
- `integrations/moonep_torch/tilexr_moonep/torch_api.py`
- Python fakes and tests under `tests/moonep/python`

### Constraints And Non-Goals

- Keep two ordered Dispatch V1 calls for hidden and optional route weights.
- Enforce stream ownership at context/workspace scope.
- Do not register Planner workspace or peer IPC memory.
- Do not release the raw owner while aligned views or async work remain.

### Acceptance And Verification

- ctypes layout matches the C header byte-for-byte.
- Context allocates `bytes + alignment - 1`, retains raw owner, and passes the
  aligned pointer/size to both Dispatch calls.
- Register happens after communicator creation; unregister precedes free and
  communicator destruction.
- Hidden/weight call presence and order are identical in fakes and real binding.
- Cross-stream reuse before quiesce and use-after-timeout are rejected.
- Existing Python unit and FFI tests pass.

### Artifacts And Interfaces

- Torch-owned registered Dispatch workspace lifecycle

## Task 9: Add End-To-End Demo, Diagnostics, And Benchmark Coverage

### Objective And Role

Provide local and future hardware evidence for the full public Dispatch path.

### Background And Prerequisites

Depends on Tasks 1 through 8.

### Modification Scope

- `tests/moonep/demo/`
- `tests/moonep/python/`
- `tools/moonep/`
- `tests/moonep/CMakeLists.txt`
- MoonEP test documentation and runners

### Constraints And Non-Goals

- No remote execution until the user approves the local implementation.
- Performance and profiling use separate runs.
- Do not claim hardware results from local Host/static tests.

### Acceptance And Verification

- Bit-exact reference validation covers FP16/BF16 hidden and FP32 weights.
- Negative `dst`, local/remote/duplicate, vector/scalar, multi-round, and
  single-rank cases are represented.
- Hot loop performs warmup 10 and 100 consecutive measured launches without
  per-round sync, allocation, registration, memset, or D2H.
- Per round, aggregation uses the maximum rank latency and reports mean, min,
  p50, p95, and max.
- Profiling parser uses adjacent timestamp differences and retains the final
  record for both payload modes.
- Hardware runners explicitly set `TILEXR_UDMA_ROUTE_POLICY=moonep_qpb` before
  communicator initialization.

### Artifacts And Interfaces

- Dispatch correctness demo and benchmark/report artifacts
- commands ready for later user-selected server execution

## Task 10: Local Verification And Review

### Objective And Role

Establish all evidence available without remote Ascend hardware and prepare a
reviewable handoff.

### Background And Prerequisites

Depends on all implementation tasks.

### Modification Scope

- no new production scope
- test/build artifacts remain untracked unless repository convention requires
  otherwise

### Constraints And Non-Goals

- Do not install CANN packages or modify system paths.
- Do not run on any server before user approval.
- Do not commit or push unless separately requested.

### Acceptance And Verification

Run the narrowest available sequence and report each evidence level separately:

1. source guards, ABI, layout, schedule, Host, UDMA registry/layout unit tests;
2. Python MoonEP unit/FFI tests;
3. CMake configure/build when the local target toolchain is available;
4. target Kernel compile only when the local CANN toolchain is available;
5. final `git diff --check`, changed-file review, forbidden reference-link scan,
   and independent Ascend C code review.

Record exact unavailable hardware/toolchain commands rather than reporting them
as passing.

### Artifacts And Interfaces

- local verification report in the final handoff
- explicit list of target-server tests awaiting user authorization

## Dependency Order

```text
Task 1
  -> Task 2
  -> Task 3
  -> Task 4
  -> Task 5
  -> Task 6
  -> Task 7
  -> Task 8
  -> Task 9
  -> Task 10
```

Tasks 2 and 3 touch disjoint implementation areas but are executed serially in
the shared workspace to simplify ABI integration and review. No subagents are
used because the active instructions require local main-agent execution unless
the user explicitly requests delegation.
