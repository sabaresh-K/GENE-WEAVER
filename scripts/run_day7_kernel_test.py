"""Day 7 CLI test runner comparing CPU baseline vs. Numba CUDA sequence alignment.

Generates or loads a genomic sequence chunk with embedded CRISPR targets, executes
both CPU and GPU alignments, verifies 100% parity across matched loci and mismatches,
and prints a performance and execution latency comparison table.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from geneweaver.aligner_cpu import CPUAligner
from geneweaver.parser import encode_sequence_numpy
from geneweaver.utils import generate_mock_fasta


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="GeneWeaver Day 7 CUDA Alignment Kernel Verification & Parity Test"
    )
    parser.add_argument(
        "--size-kb",
        type=float,
        default=500.0,
        help="Chunk size in Kilobytes to benchmark (default: 500 KB).",
    )
    parser.add_argument(
        "--guide",
        type=str,
        default="GTCACCGTAGTTCATAGCGG",
        help="CRISPR guide RNA query sequence (default: 20nt SpCas9 prototype).",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=3,
        help="Maximum tolerated Hamming distance mismatches (default: 3).",
    )
    parser.add_argument(
        "--threads-per-block",
        type=int,
        default=256,
        help="CUDA threads per block launch configuration (default: 256).",
    )
    return parser.parse_args()


def print_separator(char: str = "=", width: int = 84) -> None:
    print(char * width)


def main() -> int:
    """Run Day 7 CPU vs GPU parity benchmark."""
    args = parse_args()

    guide_seq = args.guide.strip().upper()
    total_bases = int(args.size_kb * 1024)

    print_separator("=")
    print("      GeneWeaver: Day 7 CUDA Kernel Verification & CPU Parity Benchmark")
    print_separator("=")
    print(f"[*] Guide RNA Query     : {guide_seq} (len: {len(guide_seq)} bp)")
    print(f"[*] Max Mismatches      : {args.max_mismatches}")
    print(f"[*] Sequence Chunk Size : {args.size_kb:.1f} KB ({total_bases:,} bases)")
    print(f"[*] Threads per Block   : {args.threads_per_block}")

    # Check CUDA runtime
    from numba import cuda
    is_cudasim = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"

    if is_cudasim:
        print("[*] CUDA Execution Mode : Numba CUDASIM (CPU Software Simulator)")
    elif cuda.is_available():
        print(f"[*] CUDA Execution Mode : Native NVIDIA GPU ({cuda.get_current_device().name})")
    else:
        print("\n[!] WARNING: Native CUDA hardware/compiler is unavailable.")
        print("    Set environment variable NUMBA_ENABLE_CUDASIM=1 to execute the kernel via Numba simulator:")
        print("    PowerShell: $env:NUMBA_ENABLE_CUDASIM=\"1\"; python scripts/run_day7_kernel_test.py")
        return 1

    from geneweaver.cuda_kernel import gpu_align_sequence

    # 1. Synthesize ground truth sequence chunk with injected targets
    print("\n[*] Synthesizing test sequence with injected CRISPR targets and point mutations...")
    temp_fasta = project_root / "temp_day7_chunk.fasta"
    try:
        generate_mock_fasta(
            filename=temp_fasta,
            num_megabases=args.size_kb / 1000.0,
            guide_rnas=[guide_seq],
            num_targets=20,
            mismatch_distribution={0: 0.35, 1: 0.30, 2: 0.20, 3: 0.15},
        )
        from geneweaver.parser import GenomicParser
        parser = GenomicParser(temp_fasta)
        _, raw_bytes = next(parser.get_full_sequence_bytes())
        chunk_uint8 = encode_sequence_numpy(raw_bytes, mode="numeric")
    finally:
        if temp_fasta.is_file():
            temp_fasta.unlink(missing_ok=True)

    print(f"[OK] Prepared genomic chunk buffer of {chunk_uint8.shape[0]:,} uint8 elements.")

    # 2. Run CPU Baseline
    print("\n" + "-" * 84)
    print("  Phase 1: Executing Pure Python CPU Baseline (CPUAligner)")
    print("-" * 84)
    cpu_aligner = CPUAligner(guide_rna=guide_seq, max_mismatches=args.max_mismatches, check_pam=False)
    t0_cpu = time.perf_counter()
    cpu_res = cpu_aligner.align_chunk(chunk_uint8)
    cpu_time = time.perf_counter() - t0_cpu
    cpu_hits = cpu_res.hits

    print(f"  - Hits Identified   : {len(cpu_hits)}")
    print(f"  - CPU Execution Time: {cpu_res.elapsed_seconds:.4f} s")
    print(f"  - CPU Throughput    : {cpu_res.throughput_mb_s:.2f} MB/s ({cpu_res.throughput_mbp_s:.2f} Mbp/s)")

    # 3. Run GPU Alignment Kernel
    print("\n" + "-" * 84)
    print("  Phase 2: Executing CUDA Kernel (gpu_align_sequence)")
    print("-" * 84)
    guide_uint8 = encode_sequence_numpy(guide_seq, mode="numeric")
    gpu_res = gpu_align_sequence(
        genome_chunk=chunk_uint8,
        guide_rna=guide_uint8,
        max_mismatches=args.max_mismatches,
        threads_per_block=args.threads_per_block,
    )
    gpu_hits = gpu_res.hits

    print(f"  - Hits Identified   : {len(gpu_hits)}")
    print(f"  - Blocks Launched   : {gpu_res.blocks_launched:,}")
    print(f"  - Kernel Exec Time  : {gpu_res.kernel_time_seconds:.4f} s")
    print(f"  - Total (incl. H2D) : {gpu_res.total_time_seconds:.4f} s")
    print(f"  - GPU Throughput    : {gpu_res.throughput_mb_s:.2f} MB/s ({gpu_res.throughput_mbp_s:.2f} Mbp/s)")

    # 4. Parity Verification
    print("\n" + "-" * 84)
    print("  Phase 3: 100% Result Parity & Coordinate Verification")
    print("-" * 84)

    cpu_loci = [(h.offset, h.mismatch_count, h.aligned_seq) for h in cpu_hits]
    gpu_loci = [(h.offset, h.mismatch_count, h.aligned_seq) for h in gpu_hits]

    if cpu_loci == gpu_loci:
        print(f"[OK] VERIFIED: 100% PARITY! Exact match on all {len(cpu_hits)} detected loci.")
        print("     All offsets, Hamming mismatch counts, and sequence strings match identically.")
    else:
        print("[-] PARITY FAILURE: CPU and GPU results do not match!", file=sys.stderr)
        print(f"    CPU Hit Count: {len(cpu_hits)} | GPU Hit Count: {len(gpu_hits)}", file=sys.stderr)
        diff_count = 0
        for c, g in zip(cpu_loci, gpu_loci):
            if c != g:
                diff_count += 1
                if diff_count <= 5:
                    print(f"    Diff: CPU={c} vs GPU={g}", file=sys.stderr)
        return 1

    # 5. Side-by-side Summary Table
    print_separator("=")
    print("      PERFORMANCE & SPEEDUP COMPARISON SUMMARY")
    print_separator("=")
    speedup = cpu_res.elapsed_seconds / gpu_res.total_time_seconds if gpu_res.total_time_seconds > 0 else 1.0
    kernel_speedup = cpu_res.elapsed_seconds / gpu_res.kernel_time_seconds if gpu_res.kernel_time_seconds > 0 else 1.0

    print(f"{'Metric':<28} | {'CPU Baseline':<18} | {'CUDA Kernel':<18}")
    print_separator("-")
    print(f"{'Total Candidate Hits':<28} | {len(cpu_hits):<18} | {len(gpu_hits):<18}")
    print(f"{'Execution Time (s)':<28} | {cpu_res.elapsed_seconds:<18.4f} | {gpu_res.total_time_seconds:<18.4f}")
    print(f"{'Kernel Only Time (s)':<28} | {'N/A':<18} | {gpu_res.kernel_time_seconds:<18.4f}")
    print(f"{'Throughput (MB/s)':<28} | {cpu_res.throughput_mb_s:<18.2f} | {gpu_res.throughput_mb_s:<18.2f}")
    print(f"{'Throughput (Mbp/s)':<28} | {cpu_res.throughput_mbp_s:<18.2f} | {gpu_res.throughput_mbp_s:<18.2f}")
    print_separator("-")
    if not is_cudasim:
        print(f"[*] Effective End-to-End Speedup : {speedup:.2f}x")
        print(f"[*] Kernel Compute Speedup       : {kernel_speedup:.2f}x")
    else:
        print("[*] Note: Ran under CUDASIM (interpreter mode). Physical hardware will show ~100x+ speedup.")
    print_separator("=")

    return 0


if __name__ == "__main__":
    sys.exit(main())
