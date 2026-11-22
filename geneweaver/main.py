"""GeneWeaver CLI Runner.

Pipelines genomic FASTA parsing, streaming chunking (NumPy uint8 buffers),
and CPU baseline alignment with metrics reporting (execution latency, throughput in MB/s and Mbp/s).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

# Ensure project root is in sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from geneweaver.aligner_cpu import CPUAligner, CPUAlignmentHit
from geneweaver.parser import GenomicParser
from geneweaver.utils import generate_mock_fasta


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="GeneWeaver: High-Performance CRISPR Alignment Engine (CPU Baseline Runner)"
    )
    parser.add_argument(
        "--fasta",
        type=str,
        default="mock_genome_5mb.fasta",
        help="Path to genomic FASTA file (default: mock_genome_5mb.fasta).",
    )
    parser.add_argument(
        "--guide",
        type=str,
        default="GTCACCGTAGTTCATAGCGG",
        help="CRISPR guide RNA query sequence (default: 20nt SpCas9 prototype).",
    )
    parser.add_argument(
        "--chunk-size-mb",
        type=float,
        default=1.0,
        help="Streaming chunk buffer size in Megabytes (default: 1.0 MB).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Chunk sequence overlap in base pairs to prevent border misses (default: 50 bp).",
    )
    parser.add_argument(
        "--max-mismatches",
        type=int,
        default=3,
        help="Maximum tolerated Hamming distance mismatches for off-target hits (default: 3).",
    )
    parser.add_argument(
        "--generate-size-mb",
        type=float,
        default=5.0,
        help="Size in MB of mock FASTA to create if input file does not exist (default: 5.0).",
    )
    parser.add_argument(
        "--no-pam-check",
        action="store_true",
        help="Disable downstream NGG PAM motif inspection.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute the GeneWeaver Day 3 CPU streaming alignment pipeline."""
    args = parse_args()

    fasta_path = Path(args.fasta)
    guide_seq = args.guide.strip().upper()
    chunk_size_bytes = int(args.chunk_size_mb * 1024 * 1024)
    overlap = max(args.overlap, len(guide_seq) - 1)
    check_pam = not args.no_pam_check

    print("=" * 75)
    print("  GeneWeaver: GPU-Accelerated CRISPR Alignment Engine (Day 3 CPU Baseline)")
    print("=" * 75)
    print(f"[*] Target Guide RNA   : {guide_seq} (length: {len(guide_seq)} bp)")
    print(f"[*] Max Mismatches     : {args.max_mismatches}")
    print(f"[*] PAM Motif Check    : {'Enabled (SpCas9 NGG)' if check_pam else 'Disabled'}")
    print(f"[*] Chunk Buffer Size  : {args.chunk_size_mb:.2f} MB ({chunk_size_bytes} bytes)")
    print(f"[*] Chunk Overlap      : {overlap} bp")

    # 1. Verification / Mock FASTA Generation
    if not fasta_path.is_file():
        print(f"\n[!] FASTA file '{fasta_path}' not found.")
        print(f"[*] Generating synthetic {args.generate_size_mb:.1f}MB chromosome dataset...")
        t0 = time.perf_counter()
        generate_mock_fasta(
            filename=fasta_path,
            num_megabases=args.generate_size_mb,
            guide_rnas=[guide_seq],
            num_targets=35,
            mismatch_distribution={0: 0.3, 1: 0.3, 2: 0.25, 3: 0.15},
        )
        gen_time = time.perf_counter() - t0
        print(f"[OK] Created '{fasta_path.name}' in {gen_time:.2f}s")
    else:
        print(f"\n[*] Using existing FASTA file: {fasta_path.resolve()}")

    file_size_mb = fasta_path.stat().st_size / (1024 * 1024)
    print(f"[*] Input File Size    : {file_size_mb:.2f} MB")

    # 2. Ingestion & Streaming NumPy uint8 Pipeline into CPUAligner
    print("\n" + "-" * 75)
    print("  Streaming Ingestion & uint8 CPU Baseline Alignment Execution")
    print("-" * 75)

    parser = GenomicParser(filepath=fasta_path, chunk_size=chunk_size_bytes)
    aligner = CPUAligner(
        guide_rna=guide_seq,
        max_mismatches=args.max_mismatches,
        check_pam=check_pam,
    )

    total_chunks = 0
    total_processed_bytes = 0
    total_hits = 0
    total_compute_time = 0.0
    all_hits: List[tuple[str, int, CPUAlignmentHit]] = []

    pipeline_start = time.perf_counter()

    for rec_id, offset, chunk_uint8 in parser.stream_chunks(
        chunk_size=chunk_size_bytes,
        overlap=overlap,
        return_format="numpy",
    ):
        total_chunks += 1
        chunk_len = len(chunk_uint8)
        total_processed_bytes += chunk_len

        # Execute CPU baseline alignment on uint8 numpy chunk
        chunk_result = aligner.align_chunk(chunk_uint8)

        total_compute_time += chunk_result.elapsed_seconds
        total_hits += len(chunk_result.hits)

        print(
            f"Chunk {total_chunks:02d} [{rec_id} @ offset {offset:8d}] : "
            f"Size: {chunk_len / 1024:7.1f} KB | "
            f"Hits: {len(chunk_result.hits):2d} | "
            f"Time: {chunk_result.elapsed_seconds:.3f}s | "
            f"Throughput: {chunk_result.throughput_mb_s:6.2f} MB/s ({chunk_result.throughput_mbp_s:6.2f} Mbp/s)"
        )

        for hit in chunk_result.hits:
            all_hits.append((rec_id, offset + hit.offset, hit))

        # Show first 3 hits preview for each chunk
        for hit in chunk_result.hits[:3]:
            global_locus = offset + hit.offset
            pam_status = f"PAM={hit.pam_sequence} (NGG={'YES' if hit.has_pam_ngg else 'NO'})" if hit.pam_sequence else "PAM=N/A"
            print(
                f"   -> Hit at {global_locus:8d} | "
                f"Mismatch: {hit.mismatch_count} | "
                f"Seq: {hit.aligned_seq} | {pam_status}"
            )
        if len(chunk_result.hits) > 3:
            print(f"   -> ... and {len(chunk_result.hits) - 3} more hits.")

    total_pipeline_time = time.perf_counter() - pipeline_start
    effective_mb = total_processed_bytes / (1024 * 1024)
    effective_mbp = total_processed_bytes / 1_000_000.0

    overall_compute_throughput_mb_s = (
        effective_mb / total_compute_time if total_compute_time > 0 else 0.0
    )
    overall_pipeline_throughput_mb_s = (
        effective_mb / total_pipeline_time if total_pipeline_time > 0 else 0.0
    )
    overall_pipeline_throughput_mbp_s = (
        effective_mbp / total_pipeline_time if total_pipeline_time > 0 else 0.0
    )

    # 3. Final Summary Report
    print("\n" + "=" * 75)
    print("  CPU BASELINE PERFORMANCE & ALIGNMENT SUMMARY REPORT")
    print("=" * 75)
    print(f"Chunks Streamed & Processed : {total_chunks}")
    print(f"Total Sequence Examined     : {effective_mb:.2f} MB ({total_processed_bytes:,} bases)")
    print(f"Total CRISPR Hits Identified: {total_hits}")

    # Breakdown by mismatch count
    mm_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for _, _, h in all_hits:
        if h.mismatch_count in mm_counts:
            mm_counts[h.mismatch_count] += 1
    print(f"  - Exact Matches (0 mm)    : {mm_counts[0]}")
    print(f"  - 1-Mismatch Off-targets  : {mm_counts[1]}")
    print(f"  - 2-Mismatch Off-targets  : {mm_counts[2]}")
    print(f"  - 3-Mismatch Off-targets  : {mm_counts[3]}")

    print("-" * 75)
    print(f"Total CPU Alignment Time    : {total_compute_time:.4f} seconds")
    print(f"Wall-Clock Pipeline Time    : {total_pipeline_time:.4f} seconds")
    print(f"Pure CPU Search Throughput  : {overall_compute_throughput_mb_s:.2f} MB/s")
    print(f"End-to-End Throughput       : {overall_pipeline_throughput_mb_s:.2f} MB/s ({overall_pipeline_throughput_mbp_s:.2f} Mbp/s)")
    print("=" * 75)


if __name__ == "__main__":
    main()
