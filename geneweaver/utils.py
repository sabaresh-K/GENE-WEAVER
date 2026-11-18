"""Genomic utilities and synthetic FASTA dataset generator for GeneWeaver."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union


@dataclass
class InjectedTarget:
    """Metadata for an injected CRISPR target occurrence."""

    guide_name: str
    target_with_pam: str
    injected_sequence: str
    chromosome: str
    start_pos: int
    end_pos: int
    mismatches: int
    mutation_positions: List[int]


def introduce_controlled_mismatches(
    sequence: str,
    num_mismatches: int,
    preserve_pam_length: int = 3,
) -> Tuple[str, List[int]]:
    """Introduce exact number of point mutations into a spacer sequence while preserving PAM.

    Args:
        sequence: The full target string (spacer + PAM).
        num_mismatches: Number of point mutations to introduce (e.g. 0 to 3).
        preserve_pam_length: Trailing nucleotide length for PAM that should not be mutated.

    Returns:
        Tuple of (mutated_sequence, list_of_mutated_relative_indices).
    """
    nucleotides = ["A", "C", "G", "T"]
    seq_chars = list(sequence.upper())
    seq_len = len(seq_chars)

    # Spacer region indices eligible for mutation
    spacer_indices = list(range(max(0, seq_len - preserve_pam_length)))

    if num_mismatches > len(spacer_indices):
        num_mismatches = len(spacer_indices)

    if num_mismatches <= 0:
        return sequence.upper(), []

    chosen_indices = sorted(random.sample(spacer_indices, num_mismatches))
    for idx in chosen_indices:
        curr_nt = seq_chars[idx]
        choices = [nt for nt in nucleotides if nt != curr_nt]
        seq_chars[idx] = random.choice(choices)

    return "".join(seq_chars), chosen_indices


def generate_mock_fasta(
    filename: Union[str, Path],
    num_megabases: float = 5.0,
    guide_rnas: Optional[Sequence[str]] = None,
    num_targets: int = 50,
    header_id: str = "chr1_synthetic_hg38",
    line_wrap: int = 80,
    mismatch_distribution: Optional[Dict[int, float]] = None,
    default_pam: str = "CGG",
) -> Tuple[Path, List[InjectedTarget]]:
    """Generate a synthetic chromosome FASTA file with embedded CRISPR targets and controlled point mutations.

    Args:
        filename: Destination path for the synthetic .fasta file.
        num_megabases: Total length of the synthetic genome in megabases (1 MB = 1,000,000 bases).
        guide_rnas: List of CRISPR guide RNA target strings (e.g., 20bp spacers, or 23bp spacers with PAM).
                    If 20bp, canonical PAM (`default_pam`) is appended.
        num_targets: Number of CRISPR guide target occurrences to embed across the genome.
        header_id: FASTA sequence header identifier.
        line_wrap: Number of sequence characters per line (standard FASTA format).
        mismatch_distribution: Probability distribution mapping mismatch counts (e.g., {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}).
        default_pam: Canonical SpCas9 PAM motif (default: 'CGG' for NGG).

    Returns:
        Tuple of (Path to generated FASTA, List of InjectedTarget ground-truth records).
    """
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if guide_rnas is None or len(guide_rnas) == 0:
        guide_rnas = [
            "GTCACCGTAGTTCATAGCGG",  # 20nt target 1
            "ATCGGCGATCGATTACGGTA",  # 20nt target 2
            "ACGTTAGCTAGCTAGCTGAT",  # 20nt target 3
        ]

    # Normalize guides to have PAM attached if not already present
    normalized_guides: List[Tuple[str, str]] = []
    for idx, g in enumerate(guide_rnas):
        g_clean = g.strip().upper()
        name = f"gRNA_{idx + 1}"
        if len(g_clean) == 20:
            target_pam = g_clean + default_pam
        else:
            target_pam = g_clean
        normalized_guides.append((name, target_pam))

    if mismatch_distribution is None:
        # Default: 40% exact (0 mm), 30% 1-mismatch, 20% 2-mismatches, 10% 3-mismatches
        mismatch_distribution = {0: 0.40, 1: 0.30, 2: 0.20, 3: 0.10}

    dist_keys = list(mismatch_distribution.keys())
    dist_weights = list(mismatch_distribution.values())

    total_bases = int(num_megabases * 1_000_000)
    nucleotides = ("A", "C", "G", "T")

    # Generate random background nucleotide sequence
    seq_list = random.choices(nucleotides, k=total_bases)

    injected_records: List[InjectedTarget] = []

    # Pick spaced random positions to avoid overlapping target collisions
    max_target_len = max(len(t) for _, t in normalized_guides)
    occupied_positions: List[Tuple[int, int]] = []

    attempts = 0
    max_attempts = num_targets * 100

    while len(injected_records) < num_targets and attempts < max_attempts:
        attempts += 1
        guide_name, target_with_pam = random.choice(normalized_guides)
        target_len = len(target_with_pam)

        if total_bases <= target_len:
            break

        pos = random.randint(0, total_bases - target_len)

        # Check collision with existing targets
        collision = any(not (pos + target_len <= start or pos >= end) for start, end in occupied_positions)
        if collision:
            continue

        chosen_mm = random.choices(dist_keys, weights=dist_weights, k=1)[0]
        mutated_target, mutated_relative_pos = introduce_controlled_mismatches(
            target_with_pam,
            num_mismatches=chosen_mm,
            preserve_pam_length=len(default_pam),
        )

        # Splice into genome
        seq_list[pos : pos + target_len] = list(mutated_target)
        occupied_positions.append((pos, pos + target_len))

        injected_records.append(
            InjectedTarget(
                guide_name=guide_name,
                target_with_pam=target_with_pam,
                injected_sequence=mutated_target,
                chromosome=header_id,
                start_pos=pos,
                end_pos=pos + target_len,
                mismatches=chosen_mm,
                mutation_positions=[pos + r_idx for r_idx in mutated_relative_pos],
            )
        )

    # Sort ground-truth injected records by genomic start coordinates
    injected_records.sort(key=lambda x: x.start_pos)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f">{header_id} synthetic human chromosome (size: {num_megabases:.2f}MB, targets: {len(injected_records)})\n")
        # Line-wrapped output standard FASTA formatting
        chunk_accum = []
        for i in range(0, total_bases, line_wrap):
            chunk_accum.append("".join(seq_list[i : i + line_wrap]))
            if len(chunk_accum) >= 5000:
                f.write("\n".join(chunk_accum) + "\n")
                chunk_accum.clear()
        if chunk_accum:
            f.write("\n".join(chunk_accum) + "\n")

    return filepath, injected_records
