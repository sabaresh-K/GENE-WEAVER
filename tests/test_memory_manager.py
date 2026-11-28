"""Unit tests for GPUMemoryBufferPool and pooled GPU alignment execution."""

from __future__ import annotations

import os
import numpy as np
import pytest
from numba import cuda

from geneweaver.cuda_kernel import gpu_align_sequence, gpu_align_with_workspace
from geneweaver.memory_manager import GPUMemoryBufferPool
from geneweaver.parser import encode_sequence_numpy


class TestGPUMemoryManager:
    """Test suite for Day 8 GPU memory buffer pooling and transfer optimization."""

    def test_pooled_vs_standalone_alignment_parity(self):
        """Test that pooled execution produces identical results to standalone gpu_align_sequence."""
        guide = "GTCACCGTAGTTCATAGCGG"  # 20bp
        guide_uint8 = encode_sequence_numpy(guide, mode="numeric")

        # Create multi-hit sequence
        t0 = guide
        t1 = "ATCACCGTAGTTCATAGCGG"  # 1 mm
        t2 = "ATCACCGTAGTTCATAGCGA"  # 2 mm
        full_seq = "N" * 20 + t0 + "N" * 30 + t1 + "N" * 30 + t2 + "N" * 20
        chunk_uint8 = encode_sequence_numpy(full_seq, mode="numeric")

        workspace = GPUMemoryBufferPool(
            max_chunk_size=len(chunk_uint8),
            max_guide_length=len(guide_uint8),
        )

        for mm in [0, 1, 2]:
            res_standalone = gpu_align_sequence(chunk_uint8, guide_uint8, max_mismatches=mm)
            res_pooled = gpu_align_with_workspace(workspace, chunk_uint8, guide_uint8, max_mismatches=mm)

            assert len(res_standalone.hits) == len(res_pooled.hits)
            for s_hit, p_hit in zip(res_standalone.hits, res_pooled.hits):
                assert s_hit.offset == p_hit.offset
                assert s_hit.mismatch_count == p_hit.mismatch_count
                assert s_hit.aligned_seq == p_hit.aligned_seq

            assert res_pooled.timing_breakdown is not None
            assert res_pooled.timing_breakdown.h2d_time_ms >= 0.0
            assert res_pooled.timing_breakdown.kernel_time_ms >= 0.0
            assert res_pooled.timing_breakdown.d2h_time_ms >= 0.0

        workspace.cleanup()

    def test_dynamic_capacity_growth_on_larger_chunk(self):
        """Test that workspace automatically grows when a chunk exceeds initial capacity."""
        guide = "GTCACCGTAGTTCATAGCGG"
        guide_uint8 = encode_sequence_numpy(guide, mode="numeric")

        # Initial small capacity: 100 bytes
        workspace = GPUMemoryBufferPool(max_chunk_size=100, max_guide_length=20)
        assert workspace.max_chunk_size == 100

        # Deliver a 500-byte chunk mid-stream
        larger_seq = "A" * 200 + guide + "T" * 280
        larger_uint8 = encode_sequence_numpy(larger_seq, mode="numeric")

        res = gpu_align_with_workspace(workspace, larger_uint8, guide_uint8, max_mismatches=0)

        # Capacity must have grown
        assert workspace.max_chunk_size >= 500
        assert len(res.hits) == 1
        assert res.hits[0].offset == 200
        assert res.hits[0].mismatch_count == 0

        workspace.cleanup()

    def test_buffer_pool_validation_and_cleanup(self):
        """Test parameter validation, reset, and explicit resource cleanup."""
        with pytest.raises(ValueError, match="max_chunk_size must be positive"):
            GPUMemoryBufferPool(max_chunk_size=0)

        with pytest.raises(ValueError, match="max_guide_length must be positive"):
            GPUMemoryBufferPool(max_chunk_size=100, max_guide_length=-5)

        workspace = GPUMemoryBufferPool(max_chunk_size=1000)
        assert workspace.allocated_memory_mb > 0.0
        assert workspace.d_genome is not None

        workspace.cleanup()
        assert workspace.d_genome is None
        assert workspace.d_guide is None
        assert workspace.d_mismatch_out is None
        assert workspace._allocated_bytes == 0
