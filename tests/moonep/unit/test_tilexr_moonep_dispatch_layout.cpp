#include <cstdint>
#include <iostream>
#include <limits>

#include "dispatch_layout.h"
#include "dispatch_profile.h"
#include "tilexr_types.h"

namespace {

int g_failures = 0;

#define CHECK_TRUE(expr) do { if (!(expr)) { std::cerr << "CHECK_TRUE line " \
    << __LINE__ << ": " #expr << std::endl; ++g_failures; } } while (0)
#define CHECK_EQ(lhs, rhs) do { const auto lhsValue = (lhs); const auto rhsValue = (rhs); \
    if (lhsValue != rhsValue) { std::cerr << "CHECK_EQ line " << __LINE__ \
    << ": " #lhs " != " #rhs << std::endl; ++g_failures; } } while (0)

void TestReferenceShape()
{
    TileXRMoonEp::MoonEpDispatchUrmaLayout layout {};
    CHECK_EQ(TileXRMoonEp::TileXRMoonEpBuildDispatchUrmaLayout(
        128, 128, 16, 3584, 2048,
        &layout), TileXR::TILEXR_SUCCESS);
    CHECK_EQ(layout.routeCount, 2048);
    CHECK_EQ(layout.destinationCapacity, 2048);
    CHECK_EQ(layout.hidden.rowBytes, UINT64_C(7168));
    CHECK_EQ(layout.hidden.sourceBytes, UINT64_C(917504));
    CHECK_EQ(layout.hidden.scratchSlotBytes, UINT64_C(14680064));
    CHECK_EQ(layout.weight.rowBytes, UINT64_C(4));
    CHECK_EQ(layout.weight.sourceBytes, UINT64_C(8192));
    CHECK_EQ(layout.weight.scratchSlotBytes, UINT64_C(8192));
    CHECK_EQ(layout.completionFlagsBytes, UINT64_C(8192));
    CHECK_TRUE(layout.signalOffset >=
        layout.completionFlagsOffset + layout.completionFlagsBytes);
    CHECK_EQ(layout.profileBytes,
        UINT64_C(64) * sizeof(TileXRMoonEp::DispatchProfileRecord));
    CHECK_EQ(layout.dfxBytes,
        UINT64_C(64) * sizeof(TileXRMoonEp::DispatchDfxRecord));
    CHECK_TRUE(layout.commonOffset >= layout.hidden.activeDataBytes);
    CHECK_TRUE(layout.commonOffset >= layout.weight.activeDataBytes);
    CHECK_TRUE(layout.requiredBytes <= layout.totalBytes);
    CHECK_EQ(layout.totalBytes % TileXRMoonEp::kDispatchRegistrationAlignmentBytes,
        UINT64_C(0));

    TileXRMoonEp::MoonEpDispatchUrmaLayout expanded = layout;
    const uint64_t oldCommonOffset = expanded.commonOffset;
    CHECK_EQ(TileXRMoonEp::TileXRMoonEpBindDispatchUrmaWorkspace(
        layout.totalBytes + TileXRMoonEp::kDispatchRegistrationAlignmentBytes,
        &expanded), TileXR::TILEXR_SUCCESS);
    CHECK_EQ(expanded.commonOffset,
        oldCommonOffset + TileXRMoonEp::kDispatchRegistrationAlignmentBytes);
    CHECK_TRUE(TileXRMoonEp::TileXRMoonEpBindDispatchUrmaWorkspace(
        layout.totalBytes - TileXRMoonEp::kDispatchRegistrationAlignmentBytes,
        &layout) != TileXR::TILEXR_SUCCESS);
    CHECK_TRUE(TileXRMoonEp::TileXRMoonEpBindDispatchUrmaWorkspace(
        layout.totalBytes + 64U, &layout) != TileXR::TILEXR_SUCCESS);
}

void TestFailuresAndModes()
{
    TileXRMoonEp::MoonEpDispatchUrmaLayout layout {};
    CHECK_TRUE(TileXRMoonEp::TileXRMoonEpBuildDispatchUrmaLayout(
        0, 1, 1, 1, 1, &layout) !=
        TileXR::TILEXR_SUCCESS);
    CHECK_TRUE(TileXRMoonEp::TileXRMoonEpBuildDispatchUrmaLayout(
        129, 1, 1, 1, 1, &layout) !=
        TileXR::TILEXR_SUCCESS);
    CHECK_TRUE(TileXRMoonEp::TileXRMoonEpBuildDispatchUrmaLayout(1,
        std::numeric_limits<int64_t>::max(), 2, 1,
        std::numeric_limits<int64_t>::max(), &layout) != TileXR::TILEXR_SUCCESS);
    CHECK_TRUE(TileXRMoonEp::TileXRMoonEpBuildDispatchUrmaLayout(
        1, 8, 2, 64, 15, &layout) != TileXR::TILEXR_SUCCESS);
    CHECK_EQ(TileXRMoonEp::TileXRMoonEpBuildDispatchUrmaLayout(
        1, 8, 2, 64, 32, &layout),
        TileXR::TILEXR_SUCCESS);
    CHECK_EQ(layout.routeCount, 16);
    CHECK_EQ(layout.destinationCapacity, 32);
    CHECK_EQ(layout.hidden.scratchSlotBytes, UINT64_C(4096));
    CHECK_EQ(layout.weight.scratchSlotBytes, UINT64_C(128));
    CHECK_TRUE(TileXRMoonEp::TileXRMoonEpGetActiveDispatchUrmaLayout(
        layout, TileXRMoonEp::DispatchPayloadMode::Hidden) == &layout.hidden);
    CHECK_TRUE(TileXRMoonEp::TileXRMoonEpGetActiveDispatchUrmaLayout(
        layout, TileXRMoonEp::DispatchPayloadMode::RouteWeight) == &layout.weight);
}

} // namespace

int main()
{
    TestReferenceShape();
    TestFailuresAndModes();
    return g_failures == 0 ? 0 : 1;
}
