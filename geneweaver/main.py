"""GeneWeaver CLI Runner.

Links genomic FASTA ingestion, streaming chunking, and baseline CPU alignment.
Profiles throughput and latency for CRISPR target searches.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to sys.path if run directly
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from geneweaver.baseline_cpu import cpu_align_sequence
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
        "--max-mismatches",
        type=int,
        default=2,
        help="Maximum tolerated Hamming distance mismatches for off-target hits (default: 2).",
    )
    parser.add_argument(
        "--generate-size-mb",
        type=float,
        default=5.0,
        help="Size in MB of mock FASTA to create if input file does not exist (default: 5.0).",
    )
    return parser.parse_args()


def main() -> None:
    """Run the GeneWeaver CLI workflow."""
    args = parse_args()

    fasta_path = Path(args.fasta)
    guide_seq = args.guide.strip().upper()
    chunk_size_bytes = int(args.chunk_size_mb * 1024 * 1024)
    overlap = len(guide_seq) - 1

    print("=" * 70)
    print("  GeneWeaver: GPU-Accelerated CRISPR Alignment Engine (Day 1 Baseline)")
    print("=" * 70)
    print(f"[*] Target Guide RNA   : {guide_seq} (length: {len(guide_seq)} bp)")
    print(f"[*] Max Mismatches     : {args.max_mismatches}")
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
            num_targets=30,
        )
        gen_time = time.perf_counter() - t0
        print(f"[OK] Created '{fasta_path.name}' in {gen_time:.2f}s")
    else:
        print(f"\n[*] Using existing FASTA file: {fasta_path.resolve()}")

    file_size_mb = fasta_path.stat().st_size / (1024 * 1024)
    print(f"[*] Input File Size    : {file_size_mb:.2f} MB")

    # 2. Ingestion & Streaming Chunk Processing
    print("\n" + "-" * 70)
    print("  Streaming Ingestion & CPU Baseline Alignment Execution")
    print("-" * 70)

    parser = GenomicParser(filepath=fasta_path, chunk_size=chunk_size_bytes)

    total_chunks = 0
    total_processed_bytes = 0
    total_hits = 0
    total_compute_time = 0.0

    overall_start = time.perf_counter()

    for rec_id, offset, chunk_buf in parser.stream_chunks(overlap=overlap):
        total_chunks += 1
        chunk_len = len(chunk_buf)
        total_processed_bytes += chunk_len

        res = cpu_align_sequence(
            genome_chunk=chunk_buf,
            guide_rna=guide_seq,
            max_mismatches=args.max_mismatches,
        )

        total_compute_time += res.execution_time_seconds
        total_hits += len(res.hits)

        print(
            f"Chunk {total_chunks:02d} [{rec_id} @ offset {offset:8d}] : "
            f"Size: {chunk_len / 1024:.1f} KB | "
            f"Hits: {len(res.hits):2d} | "
            f"Time: {res.execution_time_seconds:.3f}s | "
            f"Throughput: {res.throughput_mb_s:6.2f} MB/s"
        )

        # Print detailed preview for first few hits of each chunk
        for hit in res.hits[:3]:
            global_locus = offset + hit.offset
            print(
                f"   -> Hit at index {global_locus:8d} | "
                f"Mismatch: {hit.hamming_distance} | "
                f"Seq: {hit.matched_sequence}"
            )
        if len(res.hits) > 3:
            print(f"   -> ... and {len(res.hits) - 3} more hits.")

    overall_elapsed = time.perf_counter() - overall_start
    overall_throughput_mb_s = (
        (total_processed_bytes / (1024 * 1024)) / overall_elapsed
        if overall_elapsed > 0
        else 0.0
    )

    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY & PERFORMANCE REPORT")
    print("=" * 70)
    print(f"Chunks Processed        : {total_chunks}")
    print(f"Total Sequence Examined : {total_processed_bytes / (1024 * 1024):.2f} MB")
    print(f"Total Matches Found     : {total_hits}")
    print(f"Total CPU Search Time   : {total_compute_time:.4f} s")
    print(f"Wall-Clock Elapsed Time : {overall_elapsed:.4f} s")
    print(f"Overall CPU Throughput  : {overall_throughput_mb_s:.2f} MB/s")
    print("=" * 70)


if __name__ == "__main__":
    main()
