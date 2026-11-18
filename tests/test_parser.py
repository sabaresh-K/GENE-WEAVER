"""Comprehensive unit tests for GenomicParser and streaming buffer alignment."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from geneweaver.parser import (
    GenomicParser,
    decode_sequence_numpy,
    encode_sequence_numpy,
)
from geneweaver.utils import generate_mock_fasta, introduce_controlled_mismatches


class TestGenomicParser:
    """Test suite for GenomicParser chunking, overlap, and memory encoding."""

    def test_multiline_fasta_handling_and_header_stripping(self, tmp_path: Path):
        """Test correct handling of multi-line FASTA records, whitespace, and header stripping."""
        fasta_path = tmp_path / "multiline_sample.fasta"
        header = ">chrX_test_sequence description info line"
        lines = [
            header,
            "acgtACGT",
            "  TGCA\n",
            "NNNNaaaa\r\n",
            "ccccGGGG",
        ]
        fasta_path.write_text("\n".join(lines), encoding="utf-8")

        expected_sequence = "ACGTACGTTGCANNNNAAAACCCCGGGG"
        parser = GenomicParser(filepath=fasta_path, chunk_size=1024)

        # Full sequence inspection
        records = list(parser.get_full_sequence_bytes())
        assert len(records) == 1
        rec_id, full_bytes = records[0]
        assert rec_id == "chrX_test_sequence"
        assert full_bytes.decode("ascii") == expected_sequence

        # Streaming chunk inspection with return_format='str'
        chunks = list(parser.stream_chunks(chunk_size=len(expected_sequence), overlap=0, return_format="str"))
        assert len(chunks) == 1
        assert chunks[0][0] == "chrX_test_sequence"
        assert chunks[0][1] == 0
        assert chunks[0][2] == expected_sequence

    def test_exact_overlap_preservation_across_chunk_boundaries(self, tmp_path: Path):
        """Test that sequence overlap is exact across chunks so no target spanning boundaries is lost."""
        fasta_path = tmp_path / "boundary_overlap.fasta"
        # 100bp deterministic sequence
        seq_part1 = "A" * 40
        crispr_target = "GTCACCGTAGTTCATAGCGGCGG"  # 23bp target spanning boundary
        seq_part2 = "T" * 37
        full_seq = seq_part1 + crispr_target + seq_part2
        assert len(full_seq) == 100

        fasta_path.write_text(f">chr1_overlap\n{full_seq}\n", encoding="utf-8")

        # Chunk size 50, overlap 25 (stride = 25)
        # Target starts at index 40 and ends at index 63
        # Chunk 0: [0:50] contains full_seq[40:50]
        # Chunk 1: [25:75] contains full_seq[25:75] -> contains full target (40:63)!
        parser = GenomicParser(filepath=fasta_path)
        chunks = list(parser.stream_chunks(chunk_size=50, overlap=25, return_format="str"))

        assert len(chunks) == 4
        # Verify offsets
        assert chunks[0][1] == 0
        assert chunks[1][1] == 25
        assert chunks[2][1] == 50
        assert chunks[3][1] == 75

        # Verify exact overlap between consecutive chunks
        # chunk0[25:50] should equal chunk1[0:25]
        assert chunks[0][2][25:50] == chunks[1][2][0:25]
        # chunk1[25:50] should equal chunk2[0:25]
        assert chunks[1][2][25:50] == chunks[2][2][0:25]
        # chunk2[25:50] should equal chunk3[0:25]
        assert chunks[2][2][25:50] == chunks[3][2][0:25]

        # Verify that the 23bp CRISPR target spanning offset 40 is fully intact in Chunk 1
        assert crispr_target in chunks[1][2]
        rel_pos = chunks[1][2].find(crispr_target)
        assert chunks[1][1] + rel_pos == 40

    def test_numpy_encoding_and_shape_dimensions(self):
        """Test proper numpy array encoding, shapes, and uint8 numeric/ASCII mapping."""
        sample_seq = "ACGTNACGTU"
        # Expected numeric mapping: A=1, C=2, G=3, T=4, N=0, U=4
        expected_numeric = np.array([1, 2, 3, 4, 0, 1, 2, 3, 4, 4], dtype=np.uint8)

        # 1. Numeric encoding test
        enc_numeric = encode_sequence_numpy(sample_seq, mode="numeric")
        assert isinstance(enc_numeric, np.ndarray)
        assert enc_numeric.dtype == np.uint8
        assert enc_numeric.shape == (len(sample_seq),)
        assert np.array_equal(enc_numeric, expected_numeric)

        # 2. Decoding verification
        decoded_str = decode_sequence_numpy(enc_numeric, mode="numeric")
        assert decoded_str == "ACGTNACGTT"

        # 3. ASCII encoding test
        enc_ascii = encode_sequence_numpy(sample_seq, mode="ascii")
        assert enc_ascii.dtype == np.uint8
        assert enc_ascii.shape == (len(sample_seq),)
        assert np.array_equal(enc_ascii, np.array([ord(c) for c in sample_seq], dtype=np.uint8))
        assert decode_sequence_numpy(enc_ascii, mode="ascii") == sample_seq

    def test_stream_chunks_numpy_mode(self, tmp_path: Path):
        """Test parser stream_chunks yielding NumPy uint8 arrays directly for GPU pipeline staging."""
        fasta_path = tmp_path / "gpu_staging.fasta"
        fasta_path.write_text(">seq1\nACGTACGTNNNN\n", encoding="utf-8")

        parser = GenomicParser(filepath=fasta_path)
        chunks = list(parser.stream_chunks(chunk_size=8, overlap=2, return_format="numpy"))

        assert len(chunks) >= 2
        rec_id, offset, arr = chunks[0]
        assert rec_id == "seq1"
        assert offset == 0
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.uint8
        assert len(arr) == 8
        assert np.array_equal(arr, np.array([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.uint8))


class TestMockGenerator:
    """Test suite for the advanced mock FASTA generator with controlled point mutations."""

    def test_controlled_mismatches(self):
        """Verify that introduce_controlled_mismatches mutates spacer only and preserves PAM."""
        spacer_pam = "GTCACCGTAGTTCATAGCGGCGG"  # 20bp spacer + 3bp PAM ('CGG')
        pam = "CGG"

        for mm in [0, 1, 2, 3]:
            mutated, mutated_pos = introduce_controlled_mismatches(
                spacer_pam, num_mismatches=mm, preserve_pam_length=3
            )
            assert len(mutated) == len(spacer_pam)
            assert mutated.endswith(pam)  # PAM must be preserved
            assert len(mutated_pos) == mm

            # Verify actual mismatch count against original
            diffs = sum(1 for a, b in zip(spacer_pam, mutated) if a != b)
            assert diffs == mm

    def test_mock_fasta_generation_and_detection(self, tmp_path: Path):
        """Verify mock FASTA file generation with injected targets and ground truth metadata."""
        out_file = tmp_path / "synthetic_test.fasta"
        guides = ["GTCACCGTAGTTCATAGCGG", "ATCGGCGATCGATTACGGTA"]

        generated_path, injected_records = generate_mock_fasta(
            filename=out_file,
            num_megabases=0.02,  # 20,000 bases
            guide_rnas=guides,
            num_targets=15,
            mismatch_distribution={0: 0.5, 1: 0.3, 2: 0.2},
        )

        assert generated_path.is_file()
        assert len(injected_records) == 15

        # Ingest and verify ground truth positions exist in sequence
        parser = GenomicParser(filepath=generated_path)
        _, full_bytes = next(parser.get_full_sequence_bytes())
        seq_str = full_bytes.decode("ascii")

        for rec in injected_records:
            extracted = seq_str[rec.start_pos : rec.end_pos]
            assert extracted == rec.injected_sequence
            assert rec.mismatches in [0, 1, 2]
