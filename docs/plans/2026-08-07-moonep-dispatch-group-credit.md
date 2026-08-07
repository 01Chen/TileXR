# MoonEP Dispatch Group Credit Implementation Plan

## Goal And Scope

Implement the approved design in
`docs/specs/2026-08-07-moonep-dispatch-group-credit-design-zh.md` for the
direct-launch MoonEP Dispatch operator. Add grouped peer scheduling, dedicated
IPC ingress credits, peer-switch CQ reclamation, dual-QP logical batches, and
the detailed-DFX build switch. Preserve the current hidden payload contract,
legacy mode, scalar/deferred fallback outside grouped modes, public V1 ABI,
CANN 9.1 compatibility, and the existing 128-rank runtime limit.

The first hardware acceptance is balanced hidden 8P with `groupWidth=8` on an
authorized server. It validates group0, the new batch/CQ path, credit IPC
initialization, and repeated execution. It does not validate a multi-group
credit wait.

## Task 1: Pure Schedule, Credit, And Batch Contracts

**Objective and role:** Add host/device common helpers that define group/lane
peer mapping, AIV assignment, credit tokens and offsets, mode validation, and
final-CQE batch invariants before the Kernel uses them.

**Background and prerequisites:** The approved spec is authoritative. Existing
legacy helpers are in `src/moonep/dispatch/common`; existing unit coverage is
in `tests/moonep/unit/test_tilexr_moonep_dispatch_schedule.cpp`.

**Modification scope:** `dispatch_common.h`, `dispatch_schedule.h`,
`dispatch_wqe_batch.h`, new `dispatch_credit.h`, the dispatch schedule unit
test, source guards, and test CMake only where required.

**Constraints and non-goals:** Keep legacy helpers callable. Support grouped
widths 8 and 16. Do not add Kernel-only dependencies to pure helper headers.
Do not modify route selection or UDMA transport behavior.

**Acceptance and verification:** Unit tests prove remote-peer coverage without
self/duplicates for widths 8/16 and core counts 1..64, diameter and final-tail
handling, token bounds and ping-pong offsets, dual-QP logical capacity, and
final-CQE decisions. Build and run the focused schedule test locally.

**Artifacts and interfaces:** Stable helper APIs consumed by Host and Kernel.

## Task 2: Detailed-DFX Build Contract

**Objective and role:** Add `TILEXR_MOONEP_DISPATCH_ENABLE_DFX`, preserve
minimal correctness status when detailed DFX is off, and reject profiling-on
with DFX-off.

**Background and prerequisites:** Task 1 provides shared mode constants. The
current Kernel writes per-AIV DFX twice and aggregates records before output.

**Modification scope:** `src/moonep/CMakeLists.txt`, dispatch profile/common
headers, Kernel guarded code, source guards, and report/config code that labels
DFX availability.

**Constraints and non-goals:** Workspace and public ABI remain fixed. DFX-off
must retain timeouts, error detection, sticky status, kernel status, and output
suppression. Do not introduce a runtime DFX branch in the Kernel hot path.

**Acceptance and verification:** Static tests prove the option/default and
invalid CMake combination. Source guards prove DFX writes and aggregate loops
are conditional while minimal status remains unconditional. Target builds later
cover OFF/OFF and OFF/ON; ON/OFF must fail configuration.

**Artifacts and interfaces:** Compile-time feature flag and status feature bit.

## Task 3: Dedicated Credit IPC Lifecycle

**Objective and role:** Port only PR88's dedicated credit IPC allocation and
mapping lifecycle into the current communicator so group-credit can publish by
MTE without adding UDMA SQ producers.

**Background and prerequisites:** Task 1 fixes credit bytes and offsets. PR88
head `a36e91de11a5068adcf7696a416a95962f2f3638` is the reference. Current
communicator and UDMA changes in the dirty worktree must be preserved.

**Modification scope:** `src/include/comm_args.h`, `src/comm/tilexr_comm.h`,
`src/comm/tilexr_comm.cpp`, relevant comm/unit source guards, and fake/test ABI
fixtures.

**Constraints and non-goals:** Do not import PR88's rank-size increase or
unrelated transport changes. Allocation is default-disabled and controlled by
`TILEXR_ENABLE_CREDIT_IPC` before communicator init. Cleanup must handle partial
open failure and normal destruction. `InitThread` behavior must remain valid.

**Acceptance and verification:** Host/source tests prove CommArgs host/device
agreement, default-null pointers, allocation constants, SyncCommArgs wiring,
partial cleanup, and destructor ownership. Target integration later proves
8P mappings exist when enabled.

**Artifacts and interfaces:** `CommArgs.creditMems[]` and owned communicator
credit mappings.

## Task 4: Host And Kernel ABI Configuration

**Objective and role:** Parse peer mode and group width, validate group-credit
requirements and SQ worst-case bounds, and pass explicit configuration through
the direct-launch ABI.

**Background and prerequisites:** Tasks 1 through 3 provide helper and CommArgs
contracts.

**Modification scope:** Dispatch Host/launch files, Kernel signature, Host unit
fakes, ABI/source guards, benchmark configuration and launch scripts.

**Constraints and non-goals:** Defaults are `legacy` and width 16. Hardware
8P scripts explicitly use width 8. Kernel never reads environment variables.
Grouped modes reject unsupported vector/SQ shapes before launch when Host can
determine them. Host and Kernel argument order and size must have static guards.

**Acceptance and verification:** Host tests cover all valid modes and widths,
invalid values, missing credit mappings, group-credit requirements, target
8192-route acceptance, unsupported SQ window rejection, and default legacy.

**Artifacts and interfaces:** Explicit peer mode/group width Kernel arguments.

## Task 5: Grouped Dual-QP Kernel State Machine

**Objective and role:** Implement grouped peer ownership, stage/poll/credit/ring
ordering, one final CQE per peer/QP, receiver-driven next credit, and final peer
drain in the vector/batch path.

**Background and prerequisites:** Tasks 1 through 4 must be complete. Existing
two independent QP issue TBufs, route select, completion flags, timeout, UB reset,
and output pipeline are preserved.

**Modification scope:** Dispatch Kernel and only common helpers needed by it.

**Constraints and non-goals:** Credit publication uses MTE3 through
`creditMems`, never UDMA. Credit wait uses bounded MTE2 polling. Doorbell order
is stage next, poll previous, wait next credit, ring next. Intermediate batches
do not request completions; each QP's final signal WQE does. Grouped modes do
not enter deferred fallback after launch. DFX-off preserves minimal status.

**Acceptance and verification:** Source guards establish ordering, unique
publisher ownership, no UDMA credit PUT, one final completion decision, final
drain, and conditional DFX. Target CANN compile establishes API and UB resource
acceptance. Real hardware establishes execution ordering and correctness.

**Artifacts and interfaces:** The grouped vector/batch data path.

## Task 6: Local Verification And Diff Review

**Objective and role:** Establish all host-testable contracts before touching a
server and review the dirty-worktree diff for accidental changes.

**Background and prerequisites:** Tasks 1 through 5 complete.

**Modification scope:** Tests and documentation needed to keep exact commands
and limitations current.

**Constraints and non-goals:** Do not repair unrelated baseline changes. Do not
claim local compile as target Kernel proof.

**Acceptance and verification:** Configure/build focused tests, run schedule,
layout, Host, source guards, Comm/UDMA source guards, and relevant Python config
tests. Inspect modified paths and whitespace. Record unavailable target checks.

**Artifacts and interfaces:** Reviewable local implementation and evidence.

## Task 7: Authorized Server Sync, Build, And 8P Test

**Objective and role:** Sync only final modified files to a writable server copy,
compile with the target CANN toolchain, and run balanced hidden 8P.

**Background and prerequisites:** Local verification passes. Only
`141.61.49.223` and `141.61.49.192` are authorized. 223 uses the fixed private
key; 192 `/home/w50063650` is read-only and work must be under
`/home/c00838614`.

**Modification scope:** Server copy below `/home/c00838614`; local result
documentation. Raw CANN/DFX artifacts remain remote.

**Constraints and non-goals:** Check occupancy first. If occupied, wait 60
seconds and retry. Never kill another user's process. Use LF-safe/base64 remote
scripts and prevent PowerShell expansion of remote shell variables. Build with
`-O2 --cce-auto-sync --cce-aicore-arch=dav-c310-vec`. Test balanced hidden only,
no weight and no 16P. Set `groupWidth=8`.

**Acceptance and verification:** First run profiling-off/DFX-on correctness and
multi-round stability for group-credit, then profiling-off/DFX-off formal 8P
performance if correctness passes. Also retain legacy/group comparison when the
same harness supports it. Confirm all ranks pass and record exact build flags,
round statistics, mode, width, and the fact that 8P covers group0 only.

**Artifacts and interfaces:** Synchronized server source, build output, 8P
results, and updated local Markdown result record without downloaded raw logs.

## Key Risks And Stop Conditions

- A missing or incompatible CANN atomic/MTE overload requires target-header
  inspection; do not substitute an unverified primitive.
- Credit IPC may not be available for the selected communicator topology; fail
  before launch rather than mixing modes across ranks.
- A Kernel compile UB/resource failure requires reducing state or revisiting the
  design, not silently dropping synchronization.
- Any obvious device exception, repeated timeout, data mismatch, or server
  environment regression is reported before broadening tests.
- 8P width 8 does not execute group>0 credit wait; no multi-group hardware claim
  is permitted from this run.
