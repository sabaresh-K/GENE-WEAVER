"""Unit tests for CPUAligner and uint8 sliding-window CRISPR alignment."""

from __future__ import annotations

import numpy as np
import pytest

from geneweaver.aligner_cpu import CPUAligner, align_chunk_cpu
from geneweaver.parser import encode_sequence_numpy


class TestCPUAligner:
    """Test suite for CPUAligner matching logic, mismatches, and N handling."""

    def test_exact_zero_mismatch_detection(self):
        """Verify exact 0-mismatch match detection at precise genomic offset coordinates."""
        guide_seq = "GTCACCGTAGTTCATAGCGG"  # 20bp
        # Construct synthetic chunk with guide injected at index 15
        prefix = "AAAAACCCCCGGGGG"  # 15bp
        suffix = "TTTTTGGACGT"       # 11bp
        # With PAM "TGG" placed right after the 20bp guide (offset 15+20 = 35)
        full_seq = prefix + guide_seq + "TGG" + suffix
        chunk_uint8 = encode_sequence_numpy(full_seq, mode="numeric")

        aligner = CPUAligner(guide_rna=guide_seq, max_mismatches=0, check_pam=True)
        res = aligner.align_chunk(chunk_uint8)

        assert len(res.hits) == 1
        hit = res.hits[0]
        assert hit.offset == 15
        assert hit.mismatch_count == 0
        assert hit.aligned_seq == guide_seq
        assert hit.has_pam_ngg is True
        assert hit.pam_sequence == "TGG"

    def test_mismatch_detection_tiers_and_exclusion(self):
        """Verify detection of off-targets with 1, 2, and 3 mismatches and rejection of >3."""
        guide_seq = "ACGTACGTACGTACGTACGT"  # 20bp

        # Create windows with controlled mismatches
        # window 0: exact (0 mm)
        w0 = "ACGTACGTACGTACGTACGT"
        # window 1: 1 mismatch (1st base A -> T)
        w1 = "TCGTACGTACGTACGTACGT"
        # window 2: 2 mismatches (1st base A->T, 2nd base C->G)
        w2 = "TGGTACGTACGTACGTACGT"
        # window 3: 3 mismatches (1st base A->T, 2nd base C->G, 3rd base G->C)
        w3 = "TGCTACGTACGTACGTACGT"
        # window 4: 4 mismatches (exceeds max_mismatches=3)
        w4 = "TGCAACGTACGTACGTACGT"

        filler = "NNNNNNNNNN"  # 10 Ns spacing to avoid spurious overlap matches
        full_seq = filler.join([w0, w1, w2, w3, w4])
        chunk_uint8 = encode_sequence_numpy(full_seq, mode="numeric")

        # Test threshold max_mismatches = 1
        res_mm1 = align_chunk_cpu(chunk_uint8, guide_seq, max_mismatches=1, check_pam=False)
        assert len(res_mm1.hits) == 2
        assert [h.mismatch_count for h in res_mm1.hits] == [0, 1]

        # Test threshold max_mismatches = 2
        res_mm2 = align_chunk_cpu(chunk_uint8, guide_seq, max_mismatches=2, check_pam=False)
        assert len(res_mm2.hits) == 3
        assert [h.mismatch_count for h in res_mm2.hits] == [0, 1, 2]

        # Test threshold max_mismatches = 3
        res_mm3 = align_chunk_cpu(chunk_uint8, guide_seq, max_mismatches=3, check_pam=False)
        assert len(res_mm3.hits) == 4
        assert [h.mismatch_count for h in res_mm3.hits] == [0, 1, 2, 3]

        # Verify that w4 (4 mismatches) is strictly excluded
        for h in res_mm3.hits:
            assert h.mismatch_count <= 3
            assert h.aligned_seq != w4

    def test_clean_handling_of_ambiguous_n_bases(self):
        """Verify that ambiguous 'N' (encoded as 0) bases are treated as mismatches cleanly without crashing."""
        guide_seq = "ACGTACGTACGTACGTACGT"  # 20bp

        # Sequence entirely consisting of Ns
        poly_n = "N" * 100
        chunk_uint8 = encode_sequence_numpy(poly_n, mode="numeric")
        assert np.all(chunk_uint8 == 0)

        # Searching with max_mismatches=3 in pure Ns must yield 0 hits (20 mismatches)
        res = align_chunk_cpu(chunk_uint8, guide_seq, max_mismatches=3)
        assert len(res.hits) == 0
        assert res.total_bases_scanned == 100
        assert res.elapsed_seconds >= 0.0

        # Sequence with 1 ambiguous N in the target window
        # Target: ACGT N CGTACGTACGTACGT (index 4 replaced by N -> 1 mismatch)
        single_n_target = "ACGTNCGTACGTACGTACGT"
        res_single_n = align_chunk_cpu(single_n_target, guide_seq, max_mismatches=1)
        assert len(res_single_n.hits) == 1
        assert res_single_n.hits[0].offset == 0
        assert res_single_n.hits[0].mismatch_count == 1
        assert res_single_n.hits[0].aligned_seq == single_n_target

    def test_input_validation_and_edge_cases(self):
        """Verify robust error handling for invalid arguments and boundary chunk lengths."""
        with pytest.raises(ValueError, match="cannot be empty"):
            CPUAligner(guide_rna="")

        with pytest.raises(ValueError, match="non-negative"):
            CPUAligner(guide_rna="ACGT", max_mismatches=-1)

        # Chunk shorter than guide
        short_chunk = encode_sequence_numpy("ACGT", mode="numeric")
        res_short = align_chunk_cpu(short_chunk, guide_rna="ACGTACGTACGTACGTACGT", max_mismatches=2)
        assert len(res_short.hits) == 0
        assert res_short.total_bases_scanned == 4
