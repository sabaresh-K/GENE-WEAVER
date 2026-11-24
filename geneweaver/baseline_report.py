"""Week 1 CPU Baseline Summary and Metric Validation Report generator.

Loads saved benchmark JSON artifacts, validates metric schema and integrity,
computes aggregate baseline statistics, and sets target speedup thresholds
(e.g., 100x speedup for upcoming Numba CUDA kernel development).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


REQUIRED_METRIC_FIELDS = [
    "trial_name",
    "chunk_size_bytes",
    "chunk_size_mb",
    "max_mismatches",
    "guide_sequence",
    "total_bases_scanned",
    "candidate_hits_found",
    "elapsed_seconds",
    "throughput_mb_s",
    "throughput_bp_s",
    "peak_memory_mb",
]


@dataclass
class BaselineSummaryMetrics:
    """Aggregated baseline performance metrics across benchmark runs."""

    total_runs: int
    mean_throughput_mb_s: float
    min_throughput_mb_s: float
    max_throughput_mb_s: float
    mean_throughput_bp_s: float
    total_bases_profiled: int
    mean_peak_memory_mb: float
    target_100x_gpu_throughput_mb_s: float


def validate_benchmark_artifact(data: Dict[str, Any]) -> None:
    """Validate schema and metric integrity of a benchmark JSON artifact.

    Args:
        data: Parsed benchmark JSON data.

    Raises:
        ValueError: If required fields or run records are invalid.
    """
    if "runs" not in data or not isinstance(data["runs"], list):
        raise ValueError("Invalid benchmark artifact: missing or non-list 'runs' key.")
    if len(data["runs"]) == 0:
        raise ValueError("Invalid benchmark artifact: 'runs' list is empty.")

    for idx, run in enumerate(data["runs"]):
        missing = [f for f in REQUIRED_METRIC_FIELDS if f not in run]
        if missing:
            raise ValueError(f"Run index {idx} ('{run.get('trial_name')}') missing fields: {missing}")
        if run["elapsed_seconds"] <= 0:
            raise ValueError(f"Run index {idx} has invalid non-positive elapsed_seconds.")
        if run["total_bases_scanned"] <= 0:
            raise ValueError(f"Run index {idx} has invalid non-positive total_bases_scanned.")


def load_and_summarize_baseline(
    artifact_path: Union[str, Path] = "benchmarks/cpu_baseline.json",
    target_gpu_multiplier: float = 100.0,
) -> BaselineSummaryMetrics:
    """Load, validate, and compute aggregate metrics for the CPU baseline.

    Args:
        artifact_path: Path to cpu_baseline.json.
        target_gpu_multiplier: Expected speedup factor for GPU phase (default 100.0x).

    Returns:
        BaselineSummaryMetrics instance.
    """
    file_path = Path(artifact_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Baseline benchmark artifact not found: {file_path.resolve()}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    validate_benchmark_artifact(data)

    runs = data["runs"]
    throughputs_mb = [float(r["throughput_mb_s"]) for r in runs]
    throughputs_bp = [float(r["throughput_bp_s"]) for r in runs]
    total_bases = sum(int(r["total_bases_scanned"]) for r in runs)
    memories_mb = [float(r["peak_memory_mb"]) for r in runs]

    mean_tp_mb = sum(throughputs_mb) / len(throughputs_mb)
    min_tp_mb = min(throughputs_mb)
    max_tp_mb = max(throughputs_mb)
    mean_tp_bp = sum(throughputs_bp) / len(throughputs_bp)
    mean_mem = sum(memories_mb) / len(memories_mb)

    target_gpu_tp = mean_tp_mb * target_gpu_multiplier

    return BaselineSummaryMetrics(
        total_runs=len(runs),
        mean_throughput_mb_s=round(mean_tp_mb, 2),
        min_throughput_mb_s=round(min_tp_mb, 2),
        max_throughput_mb_s=round(max_tp_mb, 2),
        mean_throughput_bp_s=round(mean_tp_bp, 1),
        total_bases_profiled=total_bases,
        mean_peak_memory_mb=round(mean_mem, 3),
        target_100x_gpu_throughput_mb_s=round(target_gpu_tp, 2),
    )


def print_week1_baseline_report(
    artifact_path: Union[str, Path] = "benchmarks/cpu_baseline.json",
    target_gpu_multiplier: float = 100.0,
) -> BaselineSummaryMetrics:
    """Format and print the clean Week 1 CPU Baseline Report to terminal.

    Args:
        artifact_path: Path to baseline JSON.
        target_gpu_multiplier: Speedup factor for Week 2 target.

    Returns:
        BaselineSummaryMetrics instance.
    """
    summary = load_and_summarize_baseline(artifact_path, target_gpu_multiplier)

    print("=" * 78)
    print("      GeneWeaver: Week 1 CPU Baseline & Data Pipeline Report")
    print("=" * 78)
    print(f"[*] Benchmark Artifact       : {Path(artifact_path).resolve()}")
    print(f"[*] Total Benchmark Trials   : {summary.total_runs}")
    print(f"[*] Total Sequence Profiled  : {summary.total_bases_profiled:,} bp ({summary.total_bases_profiled / (1024*1024):.2f} MB)")
    print(f"[*] Mean CPU Throughput      : {summary.mean_throughput_mb_s:.2f} MB/s ({summary.mean_throughput_bp_s:,.0f} bp/s)")
    print(f"[*] Throughput Range         : {summary.min_throughput_mb_s:.2f} MB/s (min) - {summary.max_throughput_mb_s:.2f} MB/s (max)")
    print(f"[*] Mean Peak Memory Overhead: {summary.mean_peak_memory_mb:.3f} MB")
    print("-" * 78)
    print(f"[*] Week 2 Target ({target_gpu_multiplier:.0f}x GPU)  : >= {summary.target_100x_gpu_throughput_mb_s:.2f} MB/s")
    print("=" * 78)

    return summary


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/cpu_baseline.json"
    print_week1_baseline_report(path)
