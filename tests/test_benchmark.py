"""Unit tests for CPUBenchmark and cProfile hotspot profiling utilities."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from geneweaver.benchmark import CPUBenchmark
from geneweaver.parser import encode_sequence_numpy
from geneweaver.profiler import profile_alignment_hotspots


class TestDay4BenchmarkingAndProfiling:
    """Test suite for Day 4 benchmarking and profiling modules."""

    def test_cpu_benchmark_single_run_and_metrics(self):
        """Test single benchmark run execution, throughput, and memory tracking."""
        guide = "GTCACCGTAGTTCATAGCGG"
        benchmark = CPUBenchmark(guide_rna=guide)

        # 100KB test buffer with an injected exact match
        test_seq = "A" * 50_000 + guide + "CGG" + "C" * 50_000
        chunk_uint8 = encode_sequence_numpy(test_seq, mode="numeric")

        result = benchmark.run_single_benchmark(
            sequence_data=chunk_uint8,
            max_mismatches=2,
            trial_label="test_trial_100k",
            use_uint8_aligner=True,
        )

        assert result.trial_name == "test_trial_100k"
        assert result.candidate_hits_found >= 1
        assert result.total_bases_scanned == len(test_seq)
        assert result.elapsed_seconds > 0.0
        assert result.throughput_mb_s > 0.0
        assert result.throughput_bp_s > 0.0
        assert result.peak_memory_mb >= 0.0

    def test_benchmark_matrix_and_artifact_saving(self, tmp_path: Path):
        """Test benchmark matrix parameter sweep and JSON/CSV artifact generation."""
        guide = "ACGTACGTACGTACGTACGT"
        benchmark = CPUBenchmark(guide_rna=guide)

        # Small 100KB buffer for quick test sweep
        test_buf = np.ones(100_000, dtype=np.uint8)

        runs = benchmark.run_matrix(
            chunk_sizes_mb=[0.05, 0.08],
            mismatch_thresholds=[0, 1],
            genome_buffer=test_buf,
            use_uint8_aligner=True,
        )

        assert len(runs) == 4
        assert len(benchmark.results) == 4

        # Verify saving
        artifacts = benchmark.save_results(output_dir=tmp_path)
        json_file = artifacts["json"]
        csv_file = artifacts["csv"]

        assert json_file.is_file()
        assert csv_file.is_file()

        # Check JSON structure
        with open(json_file, "r", encoding="utf-8") as jf:
            data = json.load(jf)
            assert data["total_runs"] == 4
            assert len(data["runs"]) == 4
            assert data["runs"][0]["guide_sequence"] == guide

    def test_profile_alignment_hotspots(self):
        """Test cProfile hotspot analysis and report extraction."""
        guide = "GTCACCGTAGTTCATAGCGG"
        # 20,000 bp buffer
        sample_seq = "ACGT" * 5000

        report = profile_alignment_hotspots(
            sequence_data=sample_seq,
            guide_rna=guide,
            max_mismatches=2,
            top_n=5,
        )

        assert report.total_time >= 0.0
        assert report.total_calls > 0
        assert len(report.top_cumulative) > 0
        assert len(report.top_internal) > 0
        # The primary loop in align_chunk should appear in top hotspots
        function_names = [e.function_name for e in report.top_internal]
        assert "align_chunk" in function_names
