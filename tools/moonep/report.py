from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
from typing import Iterable, Mapping

from .contracts import CORRECTNESS_STAGES


def write_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, target)


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, object]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if quantile < 0.0 or quantile > 1.0:
        raise ValueError("quantile must be in [0, 1]")
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _metric_summary(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p90": percentile(values, 0.90),
        "p99": percentile(values, 0.99),
    }


def aggregate_rank_artifacts(
    case_dir: str | Path,
    *,
    world_size: int,
) -> dict[str, object]:
    root = Path(case_dir)
    rank_results = []
    rank_samples = []
    for rank in range(world_size):
        rank_dir = root / f"rank_{rank}"
        with (rank_dir / "result.json").open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        samples = read_jsonl(rank_dir / "samples.jsonl")
        if result.get("status") != "passed":
            raise RuntimeError(f"rank {rank} failed: {result.get('failure_reason', 'unknown')}")
        if int(result.get("rank", rank)) != rank:
            raise ValueError(f"rank_{rank} contains result metadata for another rank")
        rank_results.append(result)
        rank_samples.append(samples)
    counts = {len(samples) for samples in rank_samples}
    if len(counts) != 1 or not counts or next(iter(counts)) <= 0:
        raise ValueError(f"rank sample counts do not match: {sorted(counts)}")

    reference = rank_results[0]
    topology_keys = (
        "global_world_size",
        "node_count",
        "local_world_size",
        "planner_group_size",
        "lane_group_size",
        "physical_device_count",
        "ranks_per_device",
        "oversubscribed",
        "planner_block_dim",
        "planner_block_dim_source",
        "dispatch_aiv_core_count",
        "dispatch_aiv_core_count_source",
        "peer_memory_cross_node",
        "cross_node_validated",
    )
    reference_topology = {
        key: reference["topology"].get(key) for key in topology_keys
    }
    for rank, result in enumerate(rank_results[1:], start=1):
        if result.get("benchmark_kind", "flow") != reference.get(
            "benchmark_kind", "flow"
        ):
            raise ValueError(f"rank {rank} benchmark kind differs from rank 0")
        if result.get("dispatch_modes") != reference.get("dispatch_modes"):
            raise ValueError(f"rank {rank} Dispatch modes differ from rank 0")
        if result["case"] != reference["case"]:
            raise ValueError(f"rank {rank} case metadata differs from rank 0")
        if result["capabilities"] != reference["capabilities"]:
            raise ValueError(f"rank {rank} capability metadata differs from rank 0")
        topology = {key: result["topology"].get(key) for key in topology_keys}
        if topology != reference_topology:
            raise ValueError(f"rank {rank} topology metadata differs from rank 0")

    iteration_count = next(iter(counts))
    metric_names = set(rank_samples[0][0]["timings_us"])
    maxima: list[dict[str, object]] = []
    for iteration in range(iteration_count):
        for rank, samples in enumerate(rank_samples):
            if int(samples[iteration]["iteration"]) != iteration:
                raise ValueError(f"rank {rank} sample iteration ordering is invalid")
            if set(samples[iteration]["timings_us"]) != metric_names:
                raise ValueError("timing metric names differ across ranks")
        maxima.append(
            {
                "iteration": iteration,
                "timings_us": {
                    name: max(
                        float(rank_samples[rank][iteration]["timings_us"][name])
                        for rank in range(world_size)
                    )
                    for name in sorted(metric_names)
                },
            }
        )

    metrics = {
        name: _metric_summary(
            [float(sample["timings_us"][name]) for sample in maxima]
        )
        for name in sorted(metric_names)
    }
    first = reference
    capabilities = first["capabilities"]
    dispatch_only = first.get("benchmark_kind") == "dispatch_hot_loop"
    dispatch_modes = tuple(first.get(
        "dispatch_modes", ("hidden", "weight", "pair")))
    if dispatch_only:
        if not dispatch_modes or len(set(dispatch_modes)) != len(dispatch_modes):
            raise ValueError("Dispatch modes must be non-empty and contain no duplicates")
        required_dispatch_metrics = {
            f"{mode}_{kind}" for mode in dispatch_modes for kind in ("host", "kernel")
        }
        missing = required_dispatch_metrics - metric_names
        if missing:
            raise ValueError(
                f"dispatch hot-loop samples are missing timing metrics: {sorted(missing)}"
            )
    transport_correctness_valid = bool(
        capabilities.get("transport_correctness_valid", False)
    )
    oversubscribed = bool(first["topology"]["oversubscribed"])
    peer_memory_cross_node = bool(
        first["topology"].get("peer_memory_cross_node", False)
    )
    cross_node_validated = bool(
        first["topology"].get("cross_node_validated", False)
    )
    native_mask_valid = (
        (int(capabilities["stage_mask"]) & 3) == 3
        and (int(capabilities["stub_mask"]) & 3) == 0
    ) if dispatch_only else bool(capabilities["transport_performance_valid"])
    transport_performance_valid = bool(
        native_mask_valid
        and not oversubscribed
        and (not peer_memory_cross_node or cross_node_validated)
    )
    if oversubscribed:
        performance_scope = "oversubscribed_functional_only"
    elif peer_memory_cross_node and not cross_node_validated:
        performance_scope = "cross_node_functional_unvalidated"
    elif transport_performance_valid:
        performance_scope = "dispatch_native" if dispatch_only else "transport"
    elif transport_correctness_valid:
        performance_scope = "native_correctness_only"
    else:
        performance_scope = "stub_contract_only"
    validation_mode = (
        "planner_and_dispatch_bit_exact" if dispatch_only else
        ("full" if transport_performance_valid else performance_scope)
    )
    summary = {
        "schema_version": 1,
        "benchmark_kind": first.get("benchmark_kind", "flow"),
        "status": "passed",
        "case": first["case"],
        "dispatch_modes": list(dispatch_modes) if dispatch_only else None,
        "logical_world_size": world_size,
        "physical_device_count": first["topology"]["physical_device_count"],
        "ranks_per_device": first["topology"]["ranks_per_device"],
        "oversubscribed": oversubscribed,
        "planner_block_dim": first["topology"].get("planner_block_dim"),
        "planner_block_dim_source": first["topology"].get("planner_block_dim_source"),
        "dispatch_aiv_core_count": first["topology"].get("dispatch_aiv_core_count"),
        "dispatch_aiv_core_count_source": first["topology"].get(
            "dispatch_aiv_core_count_source"),
        "capabilities": capabilities,
        "transport_correctness_valid": transport_correctness_valid,
        "transport_performance_valid": transport_performance_valid,
        "performance_scope": performance_scope,
        "validation": {
            "passed": all(bool(item["validation"]["passed"]) for item in rank_results),
            "mode": validation_mode,
        },
        "cross_rank_max_samples": maxima,
        "metrics_us": metrics,
    }
    tokens_per_rank = int(first["case"]["tokens_per_rank"])
    throughput_metrics = ({mode: f"{mode}_kernel" for mode in dispatch_modes}
        if dispatch_only else {"end_to_end": "end_to_end"})
    throughput_by_mode = {}
    for mode, metric_name in throughput_metrics.items():
        throughput = []
        for sample in maxima:
            duration = float(sample["timings_us"][metric_name])
            throughput.append(
                tokens_per_rank * world_size * 1_000_000.0 / duration
                if duration > 0.0
                else 0.0
            )
        throughput_by_mode[mode] = _metric_summary(throughput)
    summary["tokens_per_second"] = (
        throughput_by_mode["pair" if "pair" in dispatch_modes else dispatch_modes[0]]
        if dispatch_only else
        throughput_by_mode["end_to_end"]
    )
    if dispatch_only:
        summary["tokens_per_second_by_mode"] = throughput_by_mode
    write_json(root / "summary.json", summary)
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("metric", "min", "max", "mean", "p50", "p95", "p90", "p99"),
        )
        writer.writeheader()
        for name, values in metrics.items():
            writer.writerow({"metric": name, **values})
    return summary


def aggregate_correctness_artifacts(
    case_dir: str | Path,
    *,
    world_size: int,
) -> dict[str, object]:
    root = Path(case_dir)
    results = []
    for rank in range(world_size):
        path = root / f"rank_{rank}" / "result.json"
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        if result.get("status") != "passed":
            raise RuntimeError(
                f"rank {rank} failed: {result.get('failure_reason', 'unknown')}"
            )
        if int(result.get("rank", rank)) != rank:
            raise ValueError(f"rank_{rank} contains result metadata for another rank")
        validation = result.get("validation")
        if not isinstance(validation, dict) or not isinstance(
            validation.get("stages"), list
        ):
            raise ValueError(f"rank {rank} has invalid correctness validation metadata")
        stage_names = tuple(stage.get("stage") for stage in validation["stages"])
        if stage_names != CORRECTNESS_STAGES:
            raise ValueError(
                f"rank {rank} must contain ordered correctness stages "
                f"{CORRECTNESS_STAGES}, got {stage_names}"
            )
        case_id = result.get("case", {}).get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"rank {rank} has invalid correctness case id")
        for stage in validation["stages"]:
            stage_path = (
                root
                / f"rank_{rank}"
                / "stages"
                / f"{case_id}.{stage['stage']}.json"
            )
            if not stage_path.is_file():
                raise FileNotFoundError(
                    f"rank {rank} is missing stage artifact {stage_path.name}"
                )
            with stage_path.open("r", encoding="utf-8") as handle:
                artifact = json.load(handle)
            if artifact.get("stage") != stage["stage"] or bool(
                artifact.get("passed")
            ) != bool(stage.get("passed")):
                raise ValueError(
                    f"rank {rank} stage artifact {stage_path.name} differs from result"
                )
        results.append(result)
    reference = results[0]
    for rank, result in enumerate(results[1:], start=1):
        if result["case"] != reference["case"]:
            raise ValueError(f"rank {rank} case metadata differs from rank 0")
        if result["mode"] != reference["mode"]:
            raise ValueError(f"rank {rank} mode differs from rank 0")
    stages = [
        {
            "stage": stage_name,
            "passed": all(
                bool(result["validation"]["stages"][stage_index]["passed"])
                for result in results
            ),
        }
        for stage_index, stage_name in enumerate(CORRECTNESS_STAGES)
    ]
    summary = {
        "schema_version": 1,
        "status": "passed" if all(stage["passed"] for stage in stages) else "failed",
        "mode": reference["mode"],
        "case": reference["case"],
        "logical_world_size": world_size,
        "performance_valid": False,
        "reference_backend": reference["validation"]["reference_backend"],
        "candidate_backend": reference["validation"].get("candidate_backend"),
        "stages": stages,
    }
    write_json(root / "summary.json", summary)
    return summary
