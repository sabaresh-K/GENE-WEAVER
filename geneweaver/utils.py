"""Genomic utilities and synthetic FASTA dataset generator for GeneWeaver."""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Sequence, Union


def _mutate_subsequence(subseq: str, mutation_rate: float = 0.1) -> str:
    """Introduce random point mutations into a DNA sequence."""
    nucleotides = ["A", "C", "G", "T"]
    chars = list(subseq)
    for i in range(len(chars)):
        if random.random() < mutation_rate:
            choices = [nt for nt in nucleotides if nt != chars[i]]
            chars[i] = random.choice(choices)
    return "".join(chars)


def generate_mock_fasta(
    filename: Union[str, Path],
    num_megabases: float = 5.0,
    guide_rnas: Sequence[str] | None = None,
    num_targets: int = 50,
    header_id: str = "chr1_synthetic_hg38",
    line_wrap: int = 80,
) -> Path:
    """Generate a synthetic chromosome FASTA file with embedded CRISPR targets and point mutations.

    Args:
        filename: Destination path for the synthetic .fasta file.
        num_megabases: Total length of the synthetic genome in megabases (1 MB = 1,000,000 bases).
        guide_rnas: Optional list of guide RNA target sequences (20nt) or with PAM (23nt, e.g. NGG).
                    Defaults to standard Cas9 SpCas1 targets if None.
        num_targets: Number of CRISPR guide target occurrences to embed across the genome.
        header_id: FASTA sequence header identifier.
        line_wrap: Number of sequence characters per line (standard FASTA format).

    Returns:
        Path to the generated FASTA file.
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if guide_rnas is None:
        guide_rnas = [
            "GTCACCGTAGTTCATAGCGG",  # 20nt target 1
            "ATCGGCGATCGATTACGGTA",  # 20nt target 2
            "ACGTTAGCTAGCTAGCTGAT",  # 20nt target 3
        ]

    total_bases = int(num_megabases * 1_000_000)
    nucleotides = ("A", "C", "G", "T")

    # Generate random background sequence
    seq_list = random.choices(nucleotides, k=total_bases)

    # Embed CRISPR targets with optional point mutations
    pam_motif = "CGG"  # canonical SpCas9 NGG PAM
    for _ in range(num_targets):
        chosen_guide = random.choice(guide_rnas)
        # Apply occasional 1-2 point mutations (e.g. 50% exact, 50% mutated off-target)
        if random.random() < 0.5:
            target_seq = _mutate_subsequence(chosen_guide, mutation_rate=0.08)
        else:
            target_seq = chosen_guide

        full_insert = target_seq + pam_motif
        insert_len = len(full_insert)
        if total_bases > insert_len:
            pos = random.randint(0, total_bases - insert_len - 1)
            seq_list[pos : pos + insert_len] = list(full_insert)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f">{header_id} synthetic human chromosome (size: {num_megabases:.2f}MB)\n")
        # Write wrapped lines for valid FASTA formatting
        chunk_accum = []
        for i in range(0, total_bases, line_wrap):
            chunk_accum.append("".join(seq_list[i : i + line_wrap]))
            if len(chunk_accum) >= 5000:
                f.write("\n".join(chunk_accum) + "\n")
                chunk_accum.clear()
        if chunk_accum:
            f.write("\n".join(chunk_accum) + "\n")

    return filepath
