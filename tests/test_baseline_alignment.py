"""Comprehensive unit and integration test suite for GeneWeaver Day 5 baseline alignment.

Validates exact matches, off-target detection tiers, edge cases, cross-boundary target
retention across chunk overlaps, and metric consistency between CPU aligners and CPUBenchmark.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from geneweaver.aligner_cpu import CPUAligner, align_chunk_cpu
from geneweaver.baseline_cpu import cpu_align_sequence
from geneweaver.benchmark import CPUBenchmark
from geneweaver.parser import GenomicParser, decode_sequence_numpy, encode_sequence_numpy
from geneweaver.utils import generate_mock_fasta


class TestBaselineAlignment:
    """Comprehensive test suite for Phase 1 CPU alignment and pipeline verification."""

    def test_exact_matches_and_off_target_detection_tiers(self):
        """Test exact matches (0 mm) and off-target detection across uint8-encoded arrays."""
        guide = "GTCACCGTAGTTCATAGCGG"  # 20bp
        guide_uint8 = encode_sequence_numpy(guide, mode="numeric")

        # 0 mismatches
        target_0mm = guide
        # 1 mismatch (5th nt C -> A)
        target_1mm = "GTCACAGTAGTTCATAGCGG"
        # 2 mismatches (5th nt C -> A, 10th nt T -> G)
        target_2mm = "GTCACAGTAGGTCATAGCGG"
        # 3 mismatches (5th nt C -> A, 10th nt T -> G, 15th nt T -> C)
        target_3mm = "GTCACAGTAGGTCACAGCGG"
        # 4 mismatches (exceeds threshold)
        target_4mm = "GTCACAGTAGGTCACAGAGG"

        spacing = "NNNNNNNNNN"  # 10 Ns spacer
        composite_seq = spacing.join([target_0mm, target_1mm, target_2mm, target_3mm, target_4mm])
        chunk_uint8 = encode_sequence_numpy(composite_seq, mode="numeric")

        aligner = CPUAligner(guide_rna=guide, max_mismatches=3, check_pam=False)
        result = aligner.align_chunk(chunk_uint8)

        # Should find exactly 4 hits (0, 1, 2, 3 mismatches), strictly excluding 4mm
        assert len(result.hits) == 4
        mismatches_found = [h.mismatch_count for h in result.hits]
        assert mismatches_found == [0, 1, 2, 3]

        # Verify decoded sequences
        assert result.hits[0].aligned_seq == target_0mm
        assert result.hits[1].aligned_seq == target_1mm
        assert result.hits[2].aligned_seq == target_2mm
        assert result.hits[3].aligned_seq == target_3mm

    def test_targets_split_across_chunk_boundaries_with_overlap(self, tmp_path: Path):
        """Verify zero missed CRISPR targets when targets span chunk borders and overlap >= guide length."""
        guide = "GTCACCGTAGTTCATAGCGG"  # 20bp
        pam = "CGG"                     # 3bp PAM
        full_target = guide + pam      # 23bp

        # Place target precisely across a 50-byte chunk boundary:
        # e.g., chunk boundary at offset 50. Target starts at offset 40 and ends at offset 63.
        prefix = "A" * 40
        suffix = "T" * 60
        chrom_seq = prefix + full_target + suffix  # total 40 + 23 + 60 = 123 bases
        fasta_file = tmp_path / "split_target.fasta"
        fasta_file.write_text(f">chr_split\n{chrom_seq}\n", encoding="utf-8")

        # Case 1: overlap = 25 (>= guide_len=20 and target_len=23)
        chunk_size = 50
        overlap = 25
        parser = GenomicParser(filepath=fasta_file)
        aligner = CPUAligner(guide_rna=guide, max_mismatches=0, check_pam=True)

        found_hits: list[tuple[int, str]] = []
        for rec_id, chunk_offset, chunk_buf in parser.stream_chunks(
            chunk_size=chunk_size, overlap=overlap, return_format="numpy"
        ):
            res = aligner.align_chunk(chunk_buf)
            for hit in res.hits:
                global_locus = chunk_offset + hit.offset
                found_hits.append((global_locus, hit.aligned_seq))

        # At least one chunk must capture the target intact at global offset 40
        assert len(found_hits) >= 1
        locus_matches = [locus for locus, seq in found_hits if locus == 40 and seq == guide]
        assert len(locus_matches) >= 1

    def test_edge_cases_empty_short_and_ambiguous_sequences(self):
        """Test edge cases: empty buffer, sequence shorter than guide, and ambiguous Ns."""
        guide = "GTCACCGTAGTTCATAGCGG"  # 20bp
        aligner = CPUAligner(guide_rna=guide, max_mismatches=3)

        # 1. Empty array
        empty_arr = np.array([], dtype=np.uint8)
        res_empty = aligner.align_chunk(empty_arr)
        assert len(res_empty.hits) == 0
        assert res_empty.total_bases_scanned == 0
        assert res_empty.elapsed_seconds == 0.0

        # 2. Sequence shorter than guide length
        short_arr = encode_sequence_numpy("GTCACCGT", mode="numeric")  # 8bp < 20bp
        res_short = aligner.align_chunk(short_arr)
        assert len(res_short.hits) == 0
        assert res_short.total_bases_scanned == 8
        assert res_short.elapsed_seconds == 0.0

        # 3. Ambiguous and unexpected characters (e.g., 'X', 'Z', 'N' mapped to 0)
        unusual_seq = "NNNNXXXXZZZZGTCACCGTAGTTCATAGCGGNNNN"
        unusual_uint8 = encode_sequence_numpy(unusual_seq, mode="numeric")
        res_unusual = aligner.align_chunk(unusual_uint8)
        assert len(res_unusual.hits) == 1
        assert res_unusual.hits[0].offset == 12
        assert res_unusual.hits[0].mismatch_count == 0

        # 4. max_mismatches = 0 vs higher thresholds
        strict_aligner = CPUAligner(guide_rna=guide, max_mismatches=0)
        permissive_aligner = CPUAligner(guide_rna=guide, max_mismatches=2)

        # 1-mismatch sequence
        mutated_seq = "ATCACCGTAGTTCATAGCGG"
        mutated_uint8 = encode_sequence_numpy(mutated_seq, mode="numeric")

        res_strict = strict_aligner.align_chunk(mutated_uint8)
        assert len(res_strict.hits) == 0

        res_permissive = permissive_aligner.align_chunk(mutated_uint8)
        assert len(res_permissive.hits) == 1
        assert res_permissive.hits[0].mismatch_count == 1

    def test_consistency_between_aligners_and_cpubenchmark(self):
        """Test consistency between CPUAligner, cpu_align_sequence, and CPUBenchmark."""
        guide = "GTCACCGTAGTTCATAGCGG"
        # 1000 bp synthetic sequence with known injected target
        # 200 + 20 + 200 + 20 + 560 = 1000
        seq_str = "A" * 200 + guide + "C" * 200 + "GTCACCGTAGGTCATAGCGG" + "T" * 560
        assert len(seq_str) == 1000

        # 1. CPUAligner (Day 3 uint8 engine)
        seq_uint8 = encode_sequence_numpy(seq_str, mode="numeric")
        aligner = CPUAligner(guide_rna=guide, max_mismatches=1, check_pam=False)
        res_uint8 = aligner.align_chunk(seq_uint8)

        # 2. cpu_align_sequence (Day 1 baseline)
        res_legacy = cpu_align_sequence(genome_chunk=seq_str, guide_rna=guide, max_mismatches=1)

        # Check hit counts match
        assert len(res_uint8.hits) == len(res_legacy.hits) == 2

        # Check offsets match
        offsets_uint8 = [h.offset for h in res_uint8.hits]
        offsets_legacy = [h.offset for h in res_legacy.hits]
        assert offsets_uint8 == offsets_legacy == [200, 420]

        # Check mismatches match
        mm_uint8 = [h.mismatch_count for h in res_uint8.hits]
        mm_legacy = [h.hamming_distance for h in res_legacy.hits]
        assert mm_uint8 == mm_legacy == [0, 1]

        # 3. CPUBenchmark
        benchmark = CPUBenchmark(guide_rna=guide)
        run_res = benchmark.run_single_benchmark(
            sequence_data=seq_uint8,
            max_mismatches=1,
            trial_label="consistency_check",
            use_uint8_aligner=True,
            check_pam=False,
        )
        assert run_res.candidate_hits_found == 2
        assert run_res.total_bases_scanned == 1000
        assert run_res.elapsed_seconds > 0.0
