from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from tools.moonep.dispatch_trace import (
    TRACE_CORE_FORMAT,
    TRACE_EVENT_FORMAT,
    TRACE_HEADER_BYTES,
    build_chrome_trace,
    build_phase_summary,
    build_trace_header,
    read_trace,
    trace_layout,
)
from tools.moonep.dispatch_hot_loop import _trace_iteration_index


def _trace_blob(*, dropped_count: int = 0) -> bytes:
    layout = trace_layout(1, 2)
    blob = bytearray(layout["trace_bytes"])
    header = build_trace_header(
        rank=3,
        payload_mode=0,
        iteration_count=1,
        active_core_count=1,
        event_capacity=2,
    )
    blob[:len(header)] = header
    struct.pack_into(
        TRACE_CORE_FORMAT,
        blob,
        TRACE_HEADER_BYTES,
        100,
        200,
        7,
        0,
        0,
        3,
        0,
        1,
        dropped_count,
        0,
        0,
        0,
    )
    struct.pack_into(
        TRACE_EVENT_FORMAT,
        blob,
        layout["event_offset"],
        110,
        130,
        64,
        0,
        3,
        4,
        0,
        1,
        0,
        1,
        0,
        0,
    )
    return bytes(blob)


class DispatchTraceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "dispatch_trace.bin"

    def tearDown(self):
        self.temporary.cleanup()

    def test_parse_summary_and_chrome(self):
        self.path.write_bytes(_trace_blob())
        trace = read_trace(self.path)
        self.assertEqual(trace["header"]["rank"], 3)
        self.assertEqual(trace["events"][0]["phase_name"], "wqe-build")
        summary = build_phase_summary([trace])
        self.assertEqual(summary["phases"]["wqe-build"], {
            "count": 1,
            "mean_us": 0.02,
            "p50_us": 0.02,
            "p95_us": 0.02,
            "max_us": 0.02,
            "bytes_total": 64,
            "p50_effective_gb_s": 3.2,
            "p95_effective_gb_s": 3.2,
            "max_effective_gb_s": 3.2,
        })
        chrome = build_chrome_trace([trace])
        self.assertEqual(chrome["traceEvents"][0]["args"]["peer"], 4)

    def test_rejects_dropped_events(self):
        self.path.write_bytes(_trace_blob(dropped_count=1))
        with self.assertRaisesRegex(ValueError, "dropped 1 events"):
            read_trace(self.path)

    def test_rejects_short_file(self):
        self.path.write_bytes(b"short")
        with self.assertRaisesRegex(ValueError, "short Dispatch trace"):
            read_trace(self.path)

    def test_last_iterations_map_to_zero_based_trace_slots(self):
        self.assertIsNone(_trace_iteration_index(89, 100, 10))
        self.assertEqual(_trace_iteration_index(90, 100, 10), 0)
        self.assertEqual(_trace_iteration_index(99, 100, 10), 9)
        self.assertIsNone(_trace_iteration_index(0, 100, 0))

if __name__ == "__main__":
    unittest.main()
