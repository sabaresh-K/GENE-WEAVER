"""CPU sequence alignment engine for GeneWeaver.

Executes sliding-window sequence alignment on NumPy uint8-encoded genomic chunks
(A=1, C=2, G=3, T=4, N=0) to establish a rigorous, pure-Python baseline for
CRISPR on-target and off-target search performance.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Union
import numpy as np

from geneweaver.parser import decode_sequence_numpy, encode_sequence_numpy


@dataclass
class CPUAlignmentHit:
    """Represents a matched genomic locus within a chunk."""

    offset: int
    mismatch_count: int
    aligned_seq: str
    has_pam_ngg: bool
    pam_sequence: Optional[str] = None


@dataclass
class CPUChunkAlignmentResult:
    """Aggregated alignment results and execution metrics for a genomic chunk."""

    hits: List[CPUAlignmentHit]
    elapsed_seconds: float
    total_bases_scanned: int
    throughput_mb_s: float
    throughput_mbp_s: float


class CPUAligner:
    """Pure Python CPU sliding-window alignment engine for uint8-encoded genomic arrays."""

    def __init__(
        self,
        guide_rna: Union[str, np.ndarray, Sequence[int]],
        max_mismatches: int = 3,
        check_pam: bool = True,
    ) -> None:
        """Initialize the CPU aligner with a query guide RNA.

        Args:
            guide_rna: 20bp spacer string, or 23bp spacer+PAM, or a uint8 NumPy array.
            max_mismatches: Maximum allowable Hamming distance (substitutions). Defaults to 3.
            check_pam: If True, evaluates whether the 3-base window following a 20bp spacer
                       satisfies the canonical SpCas9 'NGG' PAM motif (N can be any base, GG = 3, 3).

        Raises:
            ValueError: If guide_rna is empty or max_mismatches is negative.
        """
        if max_mismatches < 0:
            raise ValueError(f"max_mismatches must be non-negative, got {max_mismatches}")

        if isinstance(guide_rna, str):
            guide_clean = guide_rna.strip().upper()
            if len(guide_clean) == 0:
                raise ValueError("guide_rna cannot be empty.")
            self.guide_str = guide_clean
            self.guide_uint8 = encode_sequence_numpy(guide_clean, mode="numeric")
        elif isinstance(guide_rna, np.ndarray):
            if guide_rna.size == 0:
                raise ValueError("guide_rna array cannot be empty.")
            self.guide_uint8 = np.asarray(guide_rna, dtype=np.uint8)
            self.guide_str = decode_sequence_numpy(self.guide_uint8, mode="numeric")
        else:
            self.guide_uint8 = np.array(guide_rna, dtype=np.uint8)
            if self.guide_uint8.size == 0:
                raise ValueError("guide_rna sequence cannot be empty.")
            self.guide_str = decode_sequence_numpy(self.guide_uint8, mode="numeric")

        self.max_mismatches = max_mismatches
        self.check_pam = check_pam
        self.guide_len = len(self.guide_uint8)

    def align_chunk(
        self,
        chunk_array: Union[np.ndarray, str, bytes, bytearray],
    ) -> CPUChunkAlignmentResult:
        """Scan a genomic sequence chunk and identify all alignment hits within max_mismatches.

        Args:
            chunk_array: 1D NumPy uint8 array (A=1, C=2, G=3, T=4, N=0), or raw sequence string/bytes.

        Returns:
            CPUChunkAlignmentResult dataclass containing hits, execution time, and throughput metrics.
        """
        if isinstance(chunk_array, np.ndarray):
            if chunk_array.dtype != np.uint8:
                chunk_uint8 = chunk_array.astype(np.uint8)
            else:
                chunk_uint8 = chunk_array
        else:
            chunk_uint8 = encode_sequence_numpy(chunk_array, mode="numeric")

        chunk_len = len(chunk_uint8)
        guide_len = self.guide_len
        guide = self.guide_uint8
        max_mm = self.max_mismatches

        if chunk_len < guide_len:
            return CPUChunkAlignmentResult(
                hits=[],
                elapsed_seconds=0.0,
                total_bases_scanned=chunk_len,
                throughput_mb_s=0.0,
                throughput_mbp_s=0.0,
            )

        num_windows = chunk_len - guide_len + 1
        hits: List[CPUAlignmentHit] = []

        # Numerical constants: G = 3
        G_CODE = np.uint8(3)

        start_time = time.perf_counter()

        # Sliding window scan in pure Python
        for i in range(num_windows):
            mismatches = 0
            # Compare guide against chunk window
            for j in range(guide_len):
                if chunk_uint8[i + j] != guide[j]:
                    mismatches += 1
                    if mismatches > max_mm:
                        break

            if mismatches <= max_mm:
                # Extract aligned sequence
                window_slice = chunk_uint8[i : i + guide_len]
                aligned_str = decode_sequence_numpy(window_slice, mode="numeric")

                has_pam = False
                pam_str: Optional[str] = None

                # PAM validation: if guide is 20bp and chunk has at least 3 trailing bases
                if self.check_pam and guide_len == 20:
                    if i + 23 <= chunk_len:
                        pam_slice = chunk_uint8[i + 20 : i + 23]
                        # Canonical SpCas9 PAM: NGG -> base 1 is G (3), base 2 is G (3)
                        if pam_slice[1] == G_CODE and pam_slice[2] == G_CODE:
                            has_pam = True
                        pam_str = decode_sequence_numpy(pam_slice, mode="numeric")

                hits.append(
                    CPUAlignmentHit(
                        offset=i,
                        mismatch_count=mismatches,
                        aligned_seq=aligned_str,
                        has_pam_ngg=has_pam,
                        pam_sequence=pam_str,
                    )
                )

        elapsed_time = time.perf_counter() - start_time
        throughput_mb_s = (
            (chunk_len / (1024 * 1024)) / elapsed_time if elapsed_time > 0 else 0.0
        )
        throughput_mbp_s = (
            (chunk_len / 1_000_000.0) / elapsed_time if elapsed_time > 0 else 0.0
        )

        return CPUChunkAlignmentResult(
            hits=hits,
            elapsed_seconds=elapsed_time,
            total_bases_scanned=chunk_len,
            throughput_mb_s=throughput_mb_s,
            throughput_mbp_s=throughput_mbp_s,
        )


def align_chunk_cpu(
    chunk_array: Union[np.ndarray, str, bytes, bytearray],
    guide_rna: Union[str, np.ndarray, Sequence[int]],
    max_mismatches: int = 3,
    check_pam: bool = True,
) -> CPUChunkAlignmentResult:
    """Helper functional interface to execute CPU alignment on a genomic chunk.

    Args:
        chunk_array: 1D NumPy uint8 array (A=1, C=2, G=3, T=4, N=0) or string/bytes.
        guide_rna: Guide RNA query sequence.
        max_mismatches: Maximum Hamming distance allowable. Defaults to 3.
        check_pam: Whether to check SpCas9 NGG PAM proximity. Defaults to True.

    Returns:
        CPUChunkAlignmentResult with matched loci and timing statistics.
    """
    aligner = CPUAligner(
        guide_rna=guide_rna,
        max_mismatches=max_mismatches,
        check_pam=check_pam,
    )
    return aligner.align_chunk(chunk_array)
