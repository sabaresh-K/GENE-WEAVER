"""Unit tests for GeneWeaver Day 1 modules."""

from pathlib import Path
import pytest
from geneweaver.baseline_cpu import cpu_align_sequence
from geneweaver.parser import GenomicParser
from geneweaver.utils import generate_mock_fasta


def test_cpu_align_sequence_exact_match():
    genome = "AAACCCGTCACCGTAGTTCATAGCGGGGTTT"
    guide = "GTCACCGTAGTTCATAGCGG"
    res = cpu_align_sequence(genome, guide, max_mismatches=0)
    assert len(res.hits) == 1
    assert res.hits[0].offset == 6
    assert res.hits[0].hamming_distance == 0
    assert res.hits[0].matched_sequence == guide


def test_cpu_align_sequence_mismatches():
    # 2 mutations: 7th base and 10th base
    genome = "NNNGTCACCATAGTTAATAGCGGNNN"
    guide =  "GTCACCGTAGTTCATAGCGG"
    res_strict = cpu_align_sequence(genome, guide, max_mismatches=1)
    assert len(res_strict.hits) == 0

    res_permissive = cpu_align_sequence(genome, guide, max_mismatches=2)
    assert len(res_permissive.hits) == 1
    assert res_permissive.hits[0].offset == 3
    assert res_permissive.hits[0].hamming_distance == 2


def test_generate_and_parse_fasta(tmp_path: Path):
    fasta_file = tmp_path / "test_genome.fasta"
    guide = "ATCGGCGATCGATTACGGTA"
    generate_mock_fasta(
        filename=fasta_file,
        num_megabases=0.05,  # 50,000 bases
        guide_rnas=[guide],
        num_targets=10,
    )
    assert fasta_file.is_file()

    parser = GenomicParser(filepath=fasta_file, chunk_size=10_000)
    chunks = list(parser.stream_chunks(overlap=19))
    assert len(chunks) > 0
    assert all(isinstance(c[2], bytearray) for c in chunks)
