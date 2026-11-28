"""Day 8 CLI benchmark evaluating GPU memory buffer pooling vs. recurrent allocations.

Streams multiple genomic chunks, profiles H2D vs Kernel vs D2H latency,
demonstrates elimination of repeated cudaMalloc overhead, and outputs
structured metrics to benchmarks/day8_memory_transfer.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from geneweaver.cuda_kernel import gpu_align_sequence, gpu_align_with_workspace
from geneweaver.memory_manager import GPUMemoryBufferPool
from geneweaver.parser import encode_sequence_numpy
from geneweaver.utils import generate_mock_fasta


def parse_args() -> argparse.Namespace:
    """Parse benchmark arguments."""
    parser = argparse.ArgumentParser(
        description="GeneWeaver Day 8 GPU Memory Buffer Pooling & Transfer Benchmark"
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=5,
        help="Number of consecutive streamed chunks to benchmark (default: 5).",
    )
    parser.add_argument(
        "--chunk-size-kb",
        type=float,
        default=200.0,
        help="Size of each chunk in Kilobytes (default: 200 KB).",
    )
    parser.add_argument(
        "--guide",
        type=str,
        default="GTCACCGTAGTTCATAGCGG",
        help="CRISPR guide query sequence (default: 20nt SpCas9 target).",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=2,
        help="Maximum tolerated Hamming distance mismatches (default: 2).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="benchmarks/day8_memory_transfer.json",
        help="Destination JSON report file (default: benchmarks/day8_memory_transfer.json).",
    )
    return parser.parse_args()


def print_separator(char: str = "=", width: int = 88) -> None:
    print(char * width)


def main() -> int:
    """Run Day 8 memory transfer and buffer pooling benchmark."""
    args = parse_args()

    guide_seq = args.guide.strip().upper()
    chunk_bytes = int(args.chunk_size_kb * 1024)
    output_path = project_root / args.output_file

    print_separator("=")
    print("      GeneWeaver: Day 8 GPU Memory Pooling & Transfer Benchmark")
    print_separator("=")
    print(f"[*] Guide RNA Query     : {guide_seq} (len: {len(guide_seq)} bp)")
    print(f"[*] Streamed Chunks     : {args.num_chunks}")
    print(f"[*] Chunk Buffer Size   : {args.chunk_size_kb:.1f} KB ({chunk_bytes:,} bytes)")
    print(f"[*] Max Mismatches      : {args.max_mismatches}")
    print(f"[*] Output JSON Report  : {output_path.resolve()}")

    # Check CUDA runtime
    from numba import cuda
    is_cudasim = os.environ.get("NUMBA_ENABLE_CUDASIM") == "1"

    if is_cudasim:
        print("[*] CUDA Execution Mode : Numba CUDASIM (CPU Software Simulator)")
    elif cuda.is_available():
        print(f"[*] CUDA Execution Mode : Native NVIDIA GPU ({cuda.get_current_device().name})")
    else:
        print("\n[!] WARNING: Native CUDA hardware/compiler is unavailable.")
        print("    Run with NUMBA_ENABLE_CUDASIM=1 in PowerShell to execute benchmark:")
        print("    $env:NUMBA_ENABLE_CUDASIM=\"1\"; python scripts/run_day8_memory_benchmark.py")
        return 1

    # 1. Synthesize multi-chunk test data
    print("\n[*] Synthesizing streaming genomic dataset...")
    temp_fasta = project_root / "temp_day8_stream.fasta"
    try:
        generate_mock_fasta(
            filename=temp_fasta,
            num_megabases=(args.num_chunks * args.chunk_size_kb) / 1000.0,
            guide_rnas=[guide_seq],
            num_targets=args.num_chunks * 3,
            mismatch_distribution={0: 0.5, 1: 0.3, 2: 0.2},
        )
        from geneweaver.parser import GenomicParser
        parser = GenomicParser(temp_fasta)
        streamed_chunks = list(parser.stream_chunks(chunk_size=chunk_bytes, overlap=0, return_format="numpy"))
    finally:
        if temp_fasta.is_file():
            temp_fasta.unlink(missing_ok=True)

    print(f"[OK] Generated and chunked {len(streamed_chunks)} sequential stream buffers.")

    guide_uint8 = encode_sequence_numpy(guide_seq, mode="numeric")

    # 2. Mode A: Recurrent Memory Allocation (Baseline gpu_align_sequence)
    print("\n" + "-" * 88)
    print("  Mode A: Multi-Chunk Stream WITH Recurrent Allocations (cuda.to_device each chunk)")
    print("-" * 88)

    recurrent_results = []
    t_recurrent_total_start = time.perf_counter()

    for idx, (rec_id, offset, chunk_buf) in enumerate(streamed_chunks):
        res = gpu_align_sequence(
            genome_chunk=chunk_buf,
            guide_rna=guide_uint8,
            max_mismatches=args.max_mismatches,
        )
        recurrent_results.append(res)
        bd = res.timing_breakdown
        print(
            f"Chunk {idx + 1:02d} | Hits: {len(res.hits):2d} | "
            f"H2D: {bd.h2d_time_ms:6.2f}ms | Kernel: {bd.kernel_time_ms:6.2f}ms | "
            f"D2H: {bd.d2h_time_ms:6.2f}ms | Total: {bd.total_time_ms:6.2f}ms"
        )

    t_recurrent_elapsed = time.perf_counter() - t_recurrent_total_start

    # 3. Mode B: Pre-allocated GPUMemoryBufferPool Reuse
    print("\n" + "-" * 88)
    print("  Mode B: Multi-Chunk Stream WITH GPUMemoryBufferPool Pre-allocated Workspace")
    print("-" * 88)

    workspace = GPUMemoryBufferPool(
        max_chunk_size=chunk_bytes,
        max_guide_length=len(guide_uint8),
        use_pinned_host_memory=True,
    )
    print(f"[*] Pre-allocated Device Buffer Workspace: {workspace.allocated_memory_mb:.2f} MB")

    pooled_results = []
    t_pooled_total_start = time.perf_counter()

    for idx, (rec_id, offset, chunk_buf) in enumerate(streamed_chunks):
        res = gpu_align_with_workspace(
            workspace=workspace,
            genome_chunk=chunk_buf,
            guide_rna=guide_uint8,
            max_mismatches=args.max_mismatches,
        )
        pooled_results.append(res)
        bd = res.timing_breakdown
        print(
            f"Chunk {idx + 1:02d} | Hits: {len(res.hits):2d} | "
            f"H2D: {bd.h2d_time_ms:6.2f}ms | Kernel: {bd.kernel_time_ms:6.2f}ms | "
            f"D2H: {bd.d2h_time_ms:6.2f}ms | Total: {bd.total_time_ms:6.2f}ms"
        )

    t_pooled_elapsed = time.perf_counter() - t_pooled_total_start

    # 4. Result Validation & Comparison
    print("\n" + "-" * 88)
    print("  Parity & Efficiency Verification")
    print("-" * 88)

    recurrent_hits = [len(r.hits) for r in recurrent_results]
    pooled_hits = [len(r.hits) for r in pooled_results]
    assert recurrent_hits == pooled_hits, "Error: Hit counts diverged between modes!"
    print(f"[OK] Hit Parity: Perfect match across all {len(streamed_chunks)} chunks ({sum(pooled_hits)} total hits).")

    # Aggregate timing statistics
    rec_h2d_ms = sum(r.timing_breakdown.h2d_time_ms for r in recurrent_results)
    rec_kern_ms = sum(r.timing_breakdown.kernel_time_ms for r in recurrent_results)
    rec_d2h_ms = sum(r.timing_breakdown.d2h_time_ms for r in recurrent_results)

    pool_h2d_ms = sum(r.timing_breakdown.h2d_time_ms for r in pooled_results)
    pool_kern_ms = sum(r.timing_breakdown.kernel_time_ms for r in pooled_results)
    pool_d2h_ms = sum(r.timing_breakdown.d2h_time_ms for r in pooled_results)

    print_separator("=")
    print("      MEMORY TRANSFER & LATENCY PROFILE SUMMARY")
    print_separator("=")
    print(f"{'Metric':<32} | {'Recurrent Alloc':<20} | {'Pooled Workspace':<20}")
    print_separator("-")
    print(f"{'Total Stream Time (s)':<32} | {t_recurrent_elapsed:<20.4f} | {t_pooled_elapsed:<20.4f}")
    print(f"{'Cumulative H2D Transfer (ms)':<32} | {rec_h2d_ms:<20.2f} | {pool_h2d_ms:<20.2f}")
    print(f"{'Cumulative Kernel Compute (ms)':<32} | {rec_kern_ms:<20.2f} | {pool_kern_ms:<20.2f}")
    print(f"{'Cumulative D2H Transfer (ms)':<32} | {rec_d2h_ms:<20.2f} | {pool_d2h_ms:<20.2f}")
    print(f"{'Pre-allocated VRAM Footprint':<32} | {'Dynamic':<20} | {workspace.allocated_memory_mb:<17.2f} MB")
    print_separator("=")

    # 5. Persist JSON Artifact
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "benchmark": "Day8_GPU_Memory_Transfer_Benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cuda_execution_mode": "cudasim" if is_cudasim else "hardware",
        "num_chunks": args.num_chunks,
        "chunk_size_bytes": chunk_bytes,
        "chunk_size_kb": args.chunk_size_kb,
        "recurrent_allocation_total_sec": round(t_recurrent_elapsed, 4),
        "pooled_workspace_total_sec": round(t_pooled_elapsed, 4),
        "recurrent_breakdown_ms": {
            "h2d": round(rec_h2d_ms, 2),
            "kernel": round(rec_kern_ms, 2),
            "d2h": round(rec_d2h_ms, 2),
        },
        "pooled_breakdown_ms": {
            "h2d": round(pool_h2d_ms, 2),
            "kernel": round(pool_kern_ms, 2),
            "d2h": round(pool_d2h_ms, 2),
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)

    print(f"\n[OK] Benchmark report successfully saved to: {output_path.resolve()}")
    workspace.cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
