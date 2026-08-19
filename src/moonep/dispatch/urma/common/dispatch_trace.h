#ifndef TILEXR_MOONEP_DISPATCH_TRACE_H
#define TILEXR_MOONEP_DISPATCH_TRACE_H

#include <cstdint>

#include "dispatch_common.h"

namespace TileXRMoonEp {

#if defined(__CCE__) && defined(__CCE_IS_AICORE__)
#define TILEXR_MOONEP_TRACE_INLINE __attribute__((always_inline)) inline __aicore__
#else
#define TILEXR_MOONEP_TRACE_INLINE inline
#endif

constexpr uint32_t kDispatchTraceMarker = 0x54584454U; // TXDT
constexpr uint16_t kDispatchTraceVersion = 1U;
constexpr uint32_t kDispatchTraceHeaderBytes = 4096U;
constexpr uint32_t kDispatchTraceMaxIterations = 100U;
constexpr uint32_t kDispatchTraceMaxEventsPerCore = 4096U;
constexpr int32_t kDispatchTraceNoPeer = -1;
constexpr uint32_t kDispatchTraceNoQp = UINT32_MAX;
constexpr uint32_t kDispatchTraceNoGroup = UINT32_MAX;
constexpr uint32_t kDispatchTraceNoChunk = UINT32_MAX;
constexpr uint64_t kDispatchTraceCyclesPerUs = 1000U;

enum DispatchTracePhase : uint32_t {
    kDispatchTraceRouteLoad = 0U,
    kDispatchTraceRouteSelect = 1U,
    kDispatchTracePeerInit = 2U,
    kDispatchTraceWqeBuild = 3U,
    kDispatchTraceSqPublish = 4U,
    kDispatchTraceDoorbell = 5U,
    kDispatchTraceCqWait = 6U,
    kDispatchTraceUdmaExecute = 7U,
    kDispatchTraceCreditWaitMte2 = 8U,
    kDispatchTraceCreditPublishMte3 = 9U,
    kDispatchTraceCompletionFlagWait = 10U,
    kDispatchTraceLocalCopy = 11U,
    kDispatchTraceSyncAll = 12U,
    kDispatchTraceOutputCopy = 13U,
    kDispatchTraceFinalQuiet = 14U,
    kDispatchTracePhaseCount = 15U,
};

struct alignas(64) DispatchTraceHeader {
    uint32_t marker;
    uint16_t version;
    uint16_t headerBytes;
    uint32_t rank;
    uint32_t payloadMode;
    uint32_t iterationCount;
    uint32_t coreCount;
    uint32_t activeCoreCount;
    uint32_t eventCapacity;
    uint32_t phaseCount;
    uint32_t eventBytes;
    uint64_t cyclesPerUs;
    uint64_t traceBytes;
    uint64_t coreRecordOffset;
    uint64_t eventOffset;
    uint64_t reserved[7];
};

struct alignas(64) DispatchTraceCoreRecord {
    uint64_t beginCycle;
    uint64_t endCycle;
    uint64_t magic;
    uint32_t iteration;
    uint32_t core;
    uint32_t rank;
    uint32_t payloadMode;
    uint32_t eventCount;
    uint32_t droppedCount;
    uint32_t status;
    uint32_t reserved0;
    uint64_t reserved1;
};

struct alignas(64) DispatchTraceEvent {
    uint64_t beginCycle;
    uint64_t endCycle;
    uint64_t bytes;
    uint64_t sequence;
    uint32_t phase;
    int32_t peer;
    uint32_t qp;
    uint32_t group;
    uint32_t chunk;
    uint32_t wqeCount;
    uint32_t status;
    uint32_t reserved;
};

static_assert(sizeof(DispatchTraceHeader) == 128U,
    "Dispatch trace header ABI changed");
static_assert(sizeof(DispatchTraceCoreRecord) == 64U,
    "Dispatch trace core record ABI changed");
static_assert(sizeof(DispatchTraceEvent) == 64U,
    "Dispatch trace event ABI changed");

TILEXR_MOONEP_TRACE_INLINE bool DispatchTraceCheckedMultiply(
    uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (lhs != 0U && rhs > UINT64_MAX / lhs) {
        result = UINT64_MAX;
        return false;
    }
    result = lhs * rhs;
    return true;
}

TILEXR_MOONEP_TRACE_INLINE bool DispatchTraceCheckedAdd(
    uint64_t lhs, uint64_t rhs, uint64_t &result)
{
    if (rhs > UINT64_MAX - lhs) {
        result = UINT64_MAX;
        return false;
    }
    result = lhs + rhs;
    return true;
}

TILEXR_MOONEP_TRACE_INLINE bool DispatchTraceLayout(
    uint32_t iterationCount, uint32_t eventCapacity,
    uint64_t &coreRecordOffset, uint64_t &eventOffset, uint64_t &traceBytes)
{
    coreRecordOffset = kDispatchTraceHeaderBytes;
    eventOffset = traceBytes = UINT64_MAX;
    if (iterationCount == 0U || iterationCount > kDispatchTraceMaxIterations ||
        eventCapacity == 0U || eventCapacity > kDispatchTraceMaxEventsPerCore) {
        return false;
    }
    uint64_t coreCount = 0U;
    uint64_t coreBytes = 0U;
    uint64_t eventCount = 0U;
    uint64_t eventBytes = 0U;
    return DispatchTraceCheckedMultiply(iterationCount,
               kDispatchAivCoreCount, coreCount) &&
        DispatchTraceCheckedMultiply(coreCount,
            sizeof(DispatchTraceCoreRecord), coreBytes) &&
        DispatchTraceCheckedAdd(coreRecordOffset, coreBytes, eventOffset) &&
        DispatchTraceCheckedMultiply(coreCount, eventCapacity, eventCount) &&
        DispatchTraceCheckedMultiply(eventCount,
            sizeof(DispatchTraceEvent), eventBytes) &&
        DispatchTraceCheckedAdd(eventOffset, eventBytes, traceBytes);
}

TILEXR_MOONEP_TRACE_INLINE uint64_t DispatchTraceCoreRecordOffset(
    uint32_t iteration, uint32_t core)
{
    const uint32_t coreIndex = iteration * kDispatchAivCoreCount + core;
    return kDispatchTraceHeaderBytes + (static_cast<uint64_t>(coreIndex) << 6U);
}

TILEXR_MOONEP_TRACE_INLINE uint64_t DispatchTraceEventOffset(
    uint32_t iteration, uint32_t core, uint32_t event,
    uint32_t iterationCount, uint32_t eventCapacity)
{
    const uint32_t traceCoreCount = iterationCount * kDispatchAivCoreCount;
    const uint64_t eventBase = kDispatchTraceHeaderBytes +
        (static_cast<uint64_t>(traceCoreCount) << 6U);
    const uint32_t coreIndex = iteration * kDispatchAivCoreCount + core;
    const uint32_t eventIndex = coreIndex * eventCapacity + event;
    return eventBase + (static_cast<uint64_t>(eventIndex) << 6U);
}

} // namespace TileXRMoonEp

#undef TILEXR_MOONEP_TRACE_INLINE

#endif // TILEXR_MOONEP_DISPATCH_TRACE_H
