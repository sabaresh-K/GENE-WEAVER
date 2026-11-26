"""Unit tests verifying Day 7 Numba CUDA sequence alignment kernel and CPU parity."""

from __future__ import annotations

import os
import numpy as np
import pytest

from geneweaver.aligner_cpu import CPUAligner
from geneweaver.cuda_kernel import crispr_align_kernel, gpu_align_sequence
from geneweaver.parser import decode_sequence_numpy, encode_sequence_numpy


class TestCUDAKernelParity:
    """Test suite ensuring 100% parity between CPU aligner and CUDA kernel."""

    def test_exact_parity_across_mismatch_tiers(self):
        """Test exact parity between CPUAligner and gpu_align_sequence across 0, 1, 2, 3 mismatches."""
        guide = "GTCACCGTAGTTCATAGCGG"  # 20bp

        # Construct sequences with known mismatch counts
        # 0 mm
        m0 = "GTCACCGTAGTTCATAGCGG"
        # 1 mm
        m1 = "ATCACCGTAGTTCATAGCGG"
        # 2 mm
        m2 = "ATCACCGTAGTTCATAGCGA"
        # 3 mm
        m3 = "ATCACCGTAGTTAATAGCGA"
        # 4 mm (should be excluded)
        m4 = "ATCACCGTACTTAATAGCGA"

        spacing = "NNNNNNNNNN"  # 10 Ns
        full_seq = spacing.join([m0, m1, m2, m3, m4])
        chunk_uint8 = encode_sequence_numpy(full_seq, mode="numeric")
        guide_uint8 = encode_sequence_numpy(guide, mode="numeric")

        # Test each threshold from 0 to 3
        for mm in [0, 1, 2, 3]:
            # CPU Baseline
            cpu_aligner = CPUAligner(guide_rna=guide, max_mismatches=mm, check_pam=False)
            cpu_res = cpu_aligner.align_chunk(chunk_uint8)

            # GPU Kernel
            gpu_res = gpu_align_sequence(
                genome_chunk=chunk_uint8,
                guide_rna=guide_uint8,
                max_mismatches=mm,
                threads_per_block=64,
            )

            assert len(cpu_res.hits) == len(gpu_res.hits)
            for c_hit, g_hit in zip(cpu_res.hits, gpu_res.hits):
                assert c_hit.offset == g_hit.offset
                assert c_hit.mismatch_count == g_hit.mismatch_count
                assert c_hit.aligned_seq == g_hit.aligned_seq

    def test_boundary_hits_at_start_and_end(self):
        """Test boundary target hits at index 0 and index (len(chunk) - len(guide))."""
        guide = "ACGTACGTACGTACGTACGT"  # 20bp
        guide_uint8 = encode_sequence_numpy(guide, mode="numeric")

        # Target precisely at offset 0 and at the very end
        filler = "T" * 50
        # len = 20 + 50 + 20 = 90
        # End target starts at 70, ends at 90 (last window index = 90 - 20 = 70)
        boundary_seq = guide + filler + guide
        chunk_uint8 = encode_sequence_numpy(boundary_seq, mode="numeric")

        gpu_res = gpu_align_sequence(
            genome_chunk=chunk_uint8,
            guide_rna=guide_uint8,
            max_mismatches=0,
            threads_per_block=32,
        )

        assert len(gpu_res.hits) == 2
        assert gpu_res.hits[0].offset == 0
        assert gpu_res.hits[0].aligned_seq == guide
        assert gpu_res.hits[1].offset == 70
        assert gpu_res.hits[1].aligned_seq == guide

    def test_short_sequence_and_zero_hits_handling(self):
        """Test proper handling of chunks shorter than guide RNA or with zero hits."""
        guide = "GTCACCGTAGTTCATAGCGG"  # 20bp
        guide_uint8 = encode_sequence_numpy(guide, mode="numeric")

        # 1. Chunk shorter than guide
        short_seq = encode_sequence_numpy("GTCACCGT", mode="numeric")  # 8bp
        res_short = gpu_align_sequence(short_seq, guide_uint8, max_mismatches=2)
        assert len(res_short.hits) == 0
        assert res_short.blocks_launched == 0

        # 2. Sequence with zero matching hits
        no_match_seq = encode_sequence_numpy("A" * 100, mode="numeric")
        res_no_match = gpu_align_sequence(no_match_seq, guide_uint8, max_mismatches=2)
        assert len(res_no_match.hits) == 0
        assert res_no_match.total_bases_scanned == 100
        assert res_no_match.blocks_launched > 0

    def test_input_validation(self):
        """Test parameter validation in gpu_align_sequence."""
        guide_uint8 = encode_sequence_numpy("ACGT", mode="numeric")
        chunk_uint8 = encode_sequence_numpy("ACGTACGT", mode="numeric")

        with pytest.raises(ValueError, match="max_mismatches must be non-negative"):
            gpu_align_sequence(chunk_uint8, guide_uint8, max_mismatches=-1)

        with pytest.raises(ValueError, match="guide_rna cannot be empty"):
            gpu_align_sequence(chunk_uint8, np.array([], dtype=np.uint8))
