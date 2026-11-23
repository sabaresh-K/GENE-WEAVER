"""Systematic CPU benchmarking harness for GeneWeaver.

Executes parametrized sequence alignment benchmarks across configurable chunk sizes
and mismatch thresholds, measuring wall-clock duration, throughput (MB/s and bp/s),
total matches identified, and peak process memory via tracemalloc.
Persists structured benchmark results to JSON and CSV artifacts for GPU comparison.
"""

from __future__ import annotations

import csv
import json
import time
import tracemalloc
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import numpy as np

from geneweaver.aligner_cpu import CPUAligner, align_chunk_cpu
from geneweaver.baseline_cpu import cpu_align_sequence
from geneweaver.parser import GenomicParser, encode_sequence_numpy


@dataclass
class BenchmarkRunResult:
    """Stores performance metrics for a single benchmark trial."""

    trial_name: str
    chunk_size_bytes: int
    chunk_size_mb: float
    max_mismatches: int
    guide_sequence: str
    guide_length: int
    total_bases_scanned: int
    candidate_hits_found: int
    elapsed_seconds: float
    throughput_mb_s: float
    throughput_bp_s: float
    peak_memory_mb: float
    timestamp: str


@dataclass
class BenchmarkSuiteSummary:
    """Aggregates all trial results and environmental metadata."""

    suite_name: str
    generated_at: str
    total_runs: int
    runs: List[BenchmarkRunResult]


class CPUBenchmark:
    """Benchmark orchestrator for profiling CPU sliding-window alignment performance."""

    def __init__(
        self,
        guide_rna: str = "GTCACCGTAGTTCATAGCGG",
        fasta_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the CPU Benchmark suite.

        Args:
            guide_rna: Query CRISPR guide RNA sequence (default 20nt).
            fasta_path: Optional path to a genomic FASTA file.
        """
        self.guide_rna = guide_rna.strip().upper()
        self.fasta_path = Path(fasta_path) if fasta_path else None
        self.results: List[BenchmarkRunResult] = []

    def run_single_benchmark(
        self,
        sequence_data: Union[str, bytes, bytearray, np.ndarray],
        max_mismatches: int = 3,
        trial_label: str = "cpu_trial",
        check_pam: bool = True,
        use_uint8_aligner: bool = True,
    ) -> BenchmarkRunResult:
        """Run a single timed benchmark trial on an in-memory sequence buffer.

        Args:
            sequence_data: Genomic sequence buffer (str, bytes, bytearray, or np.ndarray).
            max_mismatches: Maximum Hamming distance threshold.
            trial_label: Identifier label for this run.
            check_pam: Whether to inspect downstream NGG PAM.
            use_uint8_aligner: If True, uses `CPUAligner` (Day 3 uint8 engine);
                               if False, uses `cpu_align_sequence` (Day 1 bytearray baseline).

        Returns:
            BenchmarkRunResult containing timing, throughput, hit count, and memory metrics.
        """
        # Determine sequence size
        if isinstance(sequence_data, np.ndarray):
            total_bases = sequence_data.size
            chunk_bytes = sequence_data.nbytes
        else:
            total_bases = len(sequence_data)
            chunk_bytes = total_bases

        tracemalloc.start()
        tracemalloc.reset_peak()

        t0 = time.perf_counter()

        if use_uint8_aligner:
            if isinstance(sequence_data, np.ndarray) and sequence_data.dtype == np.uint8:
                uint8_buf = sequence_data
            else:
                uint8_buf = encode_sequence_numpy(sequence_data, mode="numeric")

            aligner = CPUAligner(
                guide_rna=self.guide_rna,
                max_mismatches=max_mismatches,
                check_pam=check_pam,
            )
            res = aligner.align_chunk(uint8_buf)
            hits_count = len(res.hits)
        else:
            res_legacy = cpu_align_sequence(
                genome_chunk=sequence_data,
                guide_rna=self.guide_rna,
                max_mismatches=max_mismatches,
            )
            hits_count = len(res_legacy.hits)

        elapsed_time = time.perf_counter() - t0
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_memory_mb = peak_bytes / (1024 * 1024)
        size_mb = chunk_bytes / (1024 * 1024)
        throughput_mb_s = size_mb / elapsed_time if elapsed_time > 0 else 0.0
        throughput_bp_s = total_bases / elapsed_time if elapsed_time > 0 else 0.0

        run_result = BenchmarkRunResult(
            trial_name=trial_label,
            chunk_size_bytes=chunk_bytes,
            chunk_size_mb=round(size_mb, 3),
            max_mismatches=max_mismatches,
            guide_sequence=self.guide_rna,
            guide_length=len(self.guide_rna),
            total_bases_scanned=total_bases,
            candidate_hits_found=hits_count,
            elapsed_seconds=round(elapsed_time, 5),
            throughput_mb_s=round(throughput_mb_s, 2),
            throughput_bp_s=round(throughput_bp_s, 1),
            peak_memory_mb=round(peak_memory_mb, 4),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self.results.append(run_result)
        return run_result

    def run_matrix(
        self,
        chunk_sizes_mb: Sequence[float] = (0.5, 1.0, 2.0),
        mismatch_thresholds: Sequence[int] = (0, 1, 2, 3),
        genome_buffer: Optional[Union[str, np.ndarray]] = None,
        use_uint8_aligner: bool = True,
    ) -> List[BenchmarkRunResult]:
        """Execute a cross-product benchmark across varying chunk sizes and mismatch tolerances.

        Args:
            chunk_sizes_mb: Sequence of buffer sizes in Megabytes to test.
            mismatch_thresholds: Sequence of max Hamming distance thresholds.
            genome_buffer: Optional genomic sequence to slice chunks from. If None and fasta_path
                           is provided, parses from FASTA; otherwise synthesizes random bases.
            use_uint8_aligner: Engine selection.

        Returns:
            List of BenchmarkRunResult objects.
        """
        # Load or generate sufficient sequence buffer
        max_needed_bytes = int(max(chunk_sizes_mb) * 1024 * 1024)

        if genome_buffer is not None:
            full_seq = genome_buffer
        elif self.fasta_path and self.fasta_path.is_file():
            parser = GenomicParser(self.fasta_path)
            _, seq_bytes = next(parser.get_full_sequence_bytes())
            full_seq = seq_bytes
        else:
            # Generate synthetic buffer
            chars = np.array([1, 2, 3, 4], dtype=np.uint8)
            full_seq = np.random.choice(chars, size=max_needed_bytes)

        suite_runs: List[BenchmarkRunResult] = []

        for size_mb in chunk_sizes_mb:
            target_bytes = int(size_mb * 1024 * 1024)
            # Slice chunk buffer
            if isinstance(full_seq, (str, bytes, bytearray)):
                sub_chunk = full_seq[:target_bytes]
            elif isinstance(full_seq, np.ndarray):
                sub_chunk = full_seq[:target_bytes]
            else:
                sub_chunk = full_seq[:target_bytes]

            for mm in mismatch_thresholds:
                label = f"size_{size_mb:.1f}MB_mm_{mm}"
                result = self.run_single_benchmark(
                    sequence_data=sub_chunk,
                    max_mismatches=mm,
                    trial_label=label,
                    use_uint8_aligner=use_uint8_aligner,
                )
                suite_runs.append(result)

        return suite_runs

    def save_results(
        self,
        output_dir: Union[str, Path] = "benchmarks",
        json_filename: str = "cpu_baseline.json",
        csv_filename: str = "cpu_baseline.csv",
    ) -> Dict[str, Path]:
        """Persist benchmark execution results to structured JSON and CSV files.

        Args:
            output_dir: Directory where report files are stored.
            json_filename: Destination JSON file name.
            csv_filename: Destination CSV file name.

        Returns:
            Dictionary containing paths to the saved JSON and CSV files.
        """
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        json_file = out_path / json_filename
        csv_file = out_path / csv_filename

        # 1. Save JSON
        summary = BenchmarkSuiteSummary(
            suite_name="GeneWeaver_CPU_Baseline_Benchmark",
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            total_runs=len(self.results),
            runs=self.results,
        )
        with open(json_file, "w", encoding="utf-8") as jf:
            json.dump(asdict(summary), jf, indent=2)

        # 2. Save CSV
        if self.results:
            fieldnames = list(asdict(self.results[0]).keys())
            with open(csv_file, "w", newline="", encoding="utf-8") as cf:
                writer = csv.DictWriter(cf, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.results:
                    writer.writerow(asdict(r))

        return {"json": json_file, "csv": csv_file}
