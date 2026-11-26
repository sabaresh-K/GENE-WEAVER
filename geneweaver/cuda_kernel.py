"""Numba CUDA sequence alignment kernel and host dispatch wrapper for GeneWeaver.

Provides:
- `@cuda.jit` sliding-window alignment kernel comparing uint8-encoded genomic sequences
  with early mismatch-threshold exit to conserve GPU warp cycles.
- Host orchestration function `gpu_align_sequence()` handling H2D/D2H memory transfers,
  launch grid computation via `recommend_grid_config()`, synchronization, and match extraction.
- Full parity with `CPUAligner` and `cpu_align_sequence`, compatible with both physical
  NVIDIA GPUs and Numba's CUDA simulator (`NUMBA_ENABLE_CUDASIM=1`).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import numpy as np
from numba import cuda

from geneweaver.gpu_info import recommend_grid_config
from geneweaver.parser import decode_sequence_numpy, encode_sequence_numpy

# Sentinel value indicating that a sliding window exceeded the mismatch threshold
MISMATCH_SENTINEL = np.uint8(255)


@cuda.jit
def crispr_align_kernel(
    genome_chunk: np.ndarray,
    guide_rna: np.ndarray,
    max_mismatches: int,
    mismatch_out: np.ndarray,
) -> None:
    """CUDA kernel executing 1D sliding-window alignment for a CRISPR guide RNA.

    Each thread evaluates exactly one genomic window offset:
        - pos = cuda.grid(1)
        - Computes Hamming distance against guide_rna.
        - Exits early if mismatch count exceeds max_mismatches.
        - Stores mismatch count (or 255 if exceeded) in mismatch_out[pos].

    Args:
        genome_chunk: 1D uint8 array of reference genome sequence (A=1, C=2, G=3, T=4, N=0).
        guide_rna: 1D uint8 array of query guide sequence.
        max_mismatches: Maximum Hamming distance threshold.
        mismatch_out: 1D uint8 array of size (len(genome_chunk) - len(guide_rna) + 1).
    """
    pos = cuda.grid(1)
    chunk_len = genome_chunk.shape[0]
    guide_len = guide_rna.shape[0]

    # Strict bounds checking: ensure window fits entirely within chunk
    if pos <= chunk_len - guide_len:
        mismatches = 0
        for j in range(guide_len):
            if genome_chunk[pos + j] != guide_rna[j]:
                mismatches += 1
                if mismatches > max_mismatches:
                    break

        if mismatches <= max_mismatches:
            mismatch_out[pos] = mismatches
        else:
            mismatch_out[pos] = 255


@dataclass
class GPUAlignmentHit:
    """Represents a matched target identified by the GPU kernel."""

    offset: int
    mismatch_count: int
    aligned_seq: str


@dataclass
class GPUAlignmentResult:
    """Execution performance metrics and matched loci from GPU alignment."""

    hits: List[GPUAlignmentHit]
    kernel_time_seconds: float
    total_time_seconds: float
    transfer_h2d_seconds: float
    transfer_d2h_seconds: float
    total_bases_scanned: int
    throughput_mb_s: float
    throughput_mbp_s: float
    blocks_launched: int
    threads_per_block: int


def gpu_align_sequence(
    genome_chunk: Union[np.ndarray, str, bytes, bytearray],
    guide_rna: Union[np.ndarray, str, bytes, bytearray],
    max_mismatches: int = 3,
    threads_per_block: int = 256,
) -> GPUAlignmentResult:
    """Host dispatch function executing CUDA-accelerated CRISPR sequence alignment.

    Args:
        genome_chunk: Genomic sequence (uint8 np.ndarray or raw string/bytes).
        guide_rna: Query guide RNA (uint8 np.ndarray or string).
        max_mismatches: Maximum allowable Hamming distance (default 3).
        threads_per_block: Threads per CUDA block (default 256).

    Returns:
        GPUAlignmentResult containing hit records and granular timing/throughput metrics.

    Raises:
        ValueError: If guide_rna is empty or max_mismatches is negative.
    """
    if max_mismatches < 0:
        raise ValueError(f"max_mismatches must be non-negative, got {max_mismatches}")

    # 1. Normalize and encode input arrays to uint8
    if isinstance(genome_chunk, np.ndarray):
        chunk_uint8 = genome_chunk if genome_chunk.dtype == np.uint8 else genome_chunk.astype(np.uint8)
    else:
        chunk_uint8 = encode_sequence_numpy(genome_chunk, mode="numeric")

    if isinstance(guide_rna, np.ndarray):
        guide_uint8 = guide_rna if guide_rna.dtype == np.uint8 else guide_rna.astype(np.uint8)
    else:
        guide_uint8 = encode_sequence_numpy(guide_rna, mode="numeric")

    chunk_len = int(chunk_uint8.shape[0])
    guide_len = int(guide_uint8.shape[0])

    if guide_len == 0:
        raise ValueError("guide_rna cannot be empty.")

    total_start = time.perf_counter()

    # Edge case: sequence chunk shorter than guide RNA
    if chunk_len < guide_len:
        return GPUAlignmentResult(
            hits=[],
            kernel_time_seconds=0.0,
            total_time_seconds=0.0,
            transfer_h2d_seconds=0.0,
            transfer_d2h_seconds=0.0,
            total_bases_scanned=chunk_len,
            throughput_mb_s=0.0,
            throughput_mbp_s=0.0,
            blocks_launched=0,
            threads_per_block=threads_per_block,
        )

    num_windows = chunk_len - guide_len + 1

    # 2. Compute 1D Grid Launch Geometry
    grid_cfg = recommend_grid_config(
        sequence_length=chunk_len,
        threads_per_block=threads_per_block,
        guide_length=guide_len,
    )
    blocks = grid_cfg.blocks_per_grid
    threads = grid_cfg.threads_per_block

    # 3. Host-to-Device Memory Transfers
    t_h2d_start = time.perf_counter()
    d_genome = cuda.to_device(chunk_uint8)
    d_guide = cuda.to_device(guide_uint8)
    # Allocate device output buffer
    d_mismatch_out = cuda.device_array(num_windows, dtype=np.uint8)
    transfer_h2d_time = time.perf_counter() - t_h2d_start

    # 4. Kernel Execution & Device Synchronization
    t_kernel_start = time.perf_counter()
    crispr_align_kernel[blocks, threads](
        d_genome,
        d_guide,
        max_mismatches,
        d_mismatch_out,
    )
    cuda.synchronize()
    kernel_time = time.perf_counter() - t_kernel_start

    # 5. Device-to-Host Memory Transfer
    t_d2h_start = time.perf_counter()
    h_mismatch_out = d_mismatch_out.copy_to_host()
    transfer_d2h_time = time.perf_counter() - t_d2h_start

    # 6. Extract Matching Coordinates
    hits: List[GPUAlignmentHit] = []
    # Vectorized search for candidate indices where mismatch <= max_mismatches
    matching_indices = np.where(h_mismatch_out <= max_mismatches)[0]

    for offset in matching_indices:
        mm = int(h_mismatch_out[offset])
        window_slice = chunk_uint8[offset : offset + guide_len]
        aligned_str = decode_sequence_numpy(window_slice, mode="numeric")
        hits.append(
            GPUAlignmentHit(
                offset=int(offset),
                mismatch_count=mm,
                aligned_seq=aligned_str,
            )
        )

    total_time = time.perf_counter() - total_start
    size_mb = chunk_len / (1024 * 1024)
    size_mbp = chunk_len / 1_000_000.0

    throughput_mb_s = size_mb / total_time if total_time > 0 else 0.0
    throughput_mbp_s = size_mbp / total_time if total_time > 0 else 0.0

    return GPUAlignmentResult(
        hits=hits,
        kernel_time_seconds=kernel_time,
        total_time_seconds=total_time,
        transfer_h2d_seconds=transfer_h2d_time,
        transfer_d2h_seconds=transfer_d2h_time,
        total_bases_scanned=chunk_len,
        throughput_mb_s=throughput_mb_s,
        throughput_mbp_s=throughput_mbp_s,
        blocks_launched=blocks,
        threads_per_block=threads,
    )
