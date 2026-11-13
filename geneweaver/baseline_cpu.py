"""CPU baseline sequence alignment module for GeneWeaver.

Implements pure Python sliding-window naive sequence alignment for CRISPR guide RNAs,
computing Hamming distance against reference chunks and profiling execution time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Sequence, Union


@dataclass
class AlignmentHit:
    """Represents a matched genomic locus for a given guide RNA."""

    offset: int
    hamming_distance: int
    matched_sequence: str


@dataclass
class AlignmentResult:
    """Contains all alignment hits within a chunk and execution performance metrics."""

    hits: List[AlignmentHit]
    execution_time_seconds: float
    chunk_size_bytes: int
    throughput_mb_s: float


def cpu_align_sequence(
    genome_chunk: Union[str, bytes, bytearray],
    guide_rna: Union[str, bytes, bytearray],
    max_mismatches: int = 3,
) -> AlignmentResult:
    """Execute a CPU sliding-window naive search for a CRISPR guide RNA within a genomic sequence.

    Args:
        genome_chunk: The reference DNA sequence slice (str, bytes, or bytearray).
        guide_rna: The query CRISPR guide sequence (typically 20nt, or 23nt including PAM).
        max_mismatches: Maximum allowable Hamming distance (substitutions) to report as hit.

    Returns:
        AlignmentResult dataclass containing hits list, elapsed time, and throughput (MB/s).

    Raises:
        ValueError: If guide_rna is empty or longer than genome_chunk.
    """
    if isinstance(genome_chunk, str):
        target_bytes = bytearray(genome_chunk.upper(), encoding="ascii")
    elif isinstance(genome_chunk, bytes):
        target_bytes = bytearray(genome_chunk.upper())
    elif isinstance(genome_chunk, bytearray):
        target_bytes = bytearray(genome_chunk.upper())
    else:
        raise TypeError(f"Unsupported genome_chunk type: {type(genome_chunk)}")

    if isinstance(guide_rna, str):
        query_bytes = bytearray(guide_rna.upper(), encoding="ascii")
    elif isinstance(guide_rna, bytes):
        query_bytes = bytearray(guide_rna.upper())
    elif isinstance(guide_rna, bytearray):
        query_bytes = bytearray(guide_rna.upper())
    else:
        raise TypeError(f"Unsupported guide_rna type: {type(guide_rna)}")

    chunk_len = len(target_bytes)
    guide_len = len(query_bytes)

    if guide_len == 0:
        raise ValueError("guide_rna sequence must not be empty.")
    if guide_len > chunk_len:
        return AlignmentResult(
            hits=[],
            execution_time_seconds=0.0,
            chunk_size_bytes=chunk_len,
            throughput_mb_s=0.0,
        )

    hits: List[AlignmentHit] = []
    start_time = time.perf_counter()

    num_windows = chunk_len - guide_len + 1

    # Sliding window search
    for i in range(num_windows):
        mismatches = 0
        for j in range(guide_len):
            if target_bytes[i + j] != query_bytes[j]:
                mismatches += 1
                if mismatches > max_mismatches:
                    break

        if mismatches <= max_mismatches:
            subseq_str = target_bytes[i : i + guide_len].decode("ascii", errors="replace")
            hits.append(
                AlignmentHit(
                    offset=i,
                    hamming_distance=mismatches,
                    matched_sequence=subseq_str,
                )
            )

    elapsed_time = time.perf_counter() - start_time
    throughput_mb_s = (
        (chunk_len / (1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0.0
    )

    return AlignmentResult(
        hits=hits,
        execution_time_seconds=elapsed_time,
        chunk_size_bytes=chunk_len,
        throughput_mb_s=throughput_mb_s,
    )
