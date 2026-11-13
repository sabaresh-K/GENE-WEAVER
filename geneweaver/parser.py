"""Genomic FASTA parser module with stream chunking support for GeneWeaver.

Provides high-throughput sequential reading and chunking of massive genomic
sequences into contiguous bytearray buffers suitable for CPU/GPU streaming pipelines.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Union
from Bio import SeqIO


class GenomicParser:
    """Parser for genomic FASTA files supporting memory-efficient streaming and chunking."""

    def __init__(self, filepath: Union[str, Path], chunk_size: int = 1_048_576) -> None:
        """Initialize the GenomicParser.

        Args:
            filepath: Path to the FASTA file.
            chunk_size: Size of output sequence chunks in bytes/characters (default: 1MB = 1048576 bytes).

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If chunk_size is less than 1.
        """
        self.filepath = Path(filepath)
        if not self.filepath.is_file():
            raise FileNotFoundError(f"FASTA file not found: {self.filepath.resolve()}")
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}")

        self.chunk_size = chunk_size

    def stream_chunks(
        self,
        overlap: int = 0,
    ) -> Generator[tuple[str, int, bytearray], None, None]:
        """Stream sequence chunks as bytearrays across records in the FASTA file.

        Args:
            overlap: Number of overlapping characters retained between consecutive chunks
                     to avoid missing window alignments spanning chunk boundaries.
                     Must be non-negative and smaller than chunk_size.

        Yields:
            Tuple of:
                - record_id (str): Identifier of the genomic record/chromosome.
                - chunk_start_offset (int): Zero-based global character offset of this chunk.
                - chunk_bytes (bytearray): Contiguous sequence bytes in ASCII/UTF-8.

        Raises:
            ValueError: If overlap is negative or >= chunk_size.
        """
        if overlap < 0 or overlap >= self.chunk_size:
            raise ValueError(
                f"overlap must satisfy 0 <= overlap < chunk_size ({self.chunk_size}), got {overlap}"
            )

        stride = self.chunk_size - overlap

        with open(self.filepath, "r", encoding="utf-8") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq_str = str(record.seq).upper()
                seq_len = len(seq_str)

                if seq_len == 0:
                    continue

                for offset in range(0, seq_len, stride):
                    chunk_slice = seq_str[offset : offset + self.chunk_size]
                    yield (
                        record.id,
                        offset,
                        bytearray(chunk_slice, encoding="ascii"),
                    )

    def get_full_sequence_bytes(self) -> Generator[tuple[str, bytearray], None, None]:
        """Yield entire sequence records as bytearrays without chunking.

        Yields:
            Tuple of (record_id, bytearray).
        """
        with open(self.filepath, "r", encoding="utf-8") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq_str = str(record.seq).upper()
                yield (record.id, bytearray(seq_str, encoding="ascii"))
