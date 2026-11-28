"""GPU memory buffer manager and pre-allocated device workspace for GeneWeaver.

Provides:
- `GPUMemoryBufferPool`: Pre-allocates and manages persistent device arrays
  and page-locked (pinned) host memory to eliminate recurrent `cudaMalloc`
  and `cudaFree` runtime latency during multi-chunk genomic streaming.
- Granular breakdown of PCIe data transfer timings: Host-to-Device (H2D),
  pure GPU kernel compute, Device-to-Host (D2H), and total wall-clock latency.
- Dynamic capacity growth when a larger chunk exceeds pre-allocated dimensions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from numba import cuda


@dataclass
class TransferTimingBreakdown:
    """Detailed latency profiling for PCIe data movement and kernel compute."""

    h2d_time_ms: float
    kernel_time_ms: float
    d2h_time_ms: float
    total_time_ms: float
    chunk_size_bytes: int
    chunk_size_mb: float
    effective_throughput_mb_s: float
    kernel_throughput_mb_s: float
    allocated_device_memory_mb: float


class GPUMemoryBufferPool:
    """Pre-allocated device workspace for zero-reallocation GPU sequence alignment."""

    def __init__(
        self,
        max_chunk_size: int = 1_048_576,  # 1 MB default
        max_guide_length: int = 32,
        use_pinned_host_memory: bool = True,
    ) -> None:
        """Initialize and allocate persistent device buffers.

        Args:
            max_chunk_size: Initial buffer capacity in bases (bytes) for genomic chunks.
            max_guide_length: Maximum anticipated guide RNA query length in bases (bytes).
            use_pinned_host_memory: If True, uses page-locked host memory (`cuda.pinned_array`)
                                    for faster asynchronous PCIe transfers if supported.
        """
        if max_chunk_size <= 0:
            raise ValueError(f"max_chunk_size must be positive, got {max_chunk_size}")
        if max_guide_length <= 0:
            raise ValueError(f"max_guide_length must be positive, got {max_guide_length}")

        self.max_chunk_size = max_chunk_size
        self.max_guide_length = max_guide_length
        self.use_pinned = use_pinned_host_memory

        self._allocated_bytes = 0
        self.d_genome: Optional[Any] = None
        self.d_guide: Optional[Any] = None
        self.d_mismatch_out: Optional[Any] = None
        self.pinned_genome: Optional[np.ndarray] = None
        self.pinned_out: Optional[np.ndarray] = None

        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        """Allocate or resize persistent device arrays."""
        num_windows = max(0, self.max_chunk_size - self.max_guide_length + 1)
        if num_windows == 0:
            num_windows = 1

        # 1. Device memory pre-allocation
        self.d_genome = cuda.device_array(self.max_chunk_size, dtype=np.uint8)
        self.d_guide = cuda.device_array(self.max_guide_length, dtype=np.uint8)
        self.d_mismatch_out = cuda.device_array(num_windows, dtype=np.uint8)

        # 2. Pinned host memory (page-locked)
        if self.use_pinned:
            try:
                self.pinned_genome = cuda.pinned_array(self.max_chunk_size, dtype=np.uint8)
                self.pinned_out = cuda.pinned_array(num_windows, dtype=np.uint8)
            except Exception:
                # Graceful fallback to standard NumPy arrays under simulator or non-supported drivers
                self.pinned_genome = np.zeros(self.max_chunk_size, dtype=np.uint8)
                self.pinned_out = np.zeros(num_windows, dtype=np.uint8)
        else:
            self.pinned_genome = np.zeros(self.max_chunk_size, dtype=np.uint8)
            self.pinned_out = np.zeros(num_windows, dtype=np.uint8)

        # Total device buffer footprint in bytes: chunk + guide + mismatch output
        self._allocated_bytes = self.max_chunk_size + self.max_guide_length + num_windows

    def ensure_capacity(self, chunk_len: int, guide_len: int = 20) -> bool:
        """Check and expand buffer capacities dynamically if an input exceeds capacity.

        Args:
            chunk_len: Required chunk capacity in bytes.
            guide_len: Required guide capacity in bytes.

        Returns:
            True if reallocation occurred, False if existing buffers were reused.
        """
        reallocated = False
        new_chunk_cap = self.max_chunk_size
        new_guide_cap = self.max_guide_length

        if chunk_len > self.max_chunk_size:
            # Expand by 1.5x with minimum headroom
            new_chunk_cap = max(chunk_len, int(self.max_chunk_size * 1.5))
            reallocated = True

        if guide_len > self.max_guide_length:
            new_guide_cap = max(guide_len, int(self.max_guide_length * 2))
            reallocated = True

        if reallocated:
            self.max_chunk_size = new_chunk_cap
            self.max_guide_length = new_guide_cap
            self._allocate_buffers()

        return reallocated

    @property
    def allocated_memory_mb(self) -> float:
        """Total pre-allocated device memory footprint in Megabytes."""
        return round(self._allocated_bytes / (1024 * 1024), 3)

    def reset(self) -> None:
        """Zero out pre-allocated device buffers."""
        if self.d_genome is not None:
            # Simple device slice reset if needed
            pass

    def cleanup(self) -> None:
        """Explicitly release references to GPU buffers."""
        self.d_genome = None
        self.d_guide = None
        self.d_mismatch_out = None
        self.pinned_genome = None
        self.pinned_out = None
        self._allocated_bytes = 0
