"""Genomic FASTA parser module with stream chunking support for GeneWeaver.

Provides high-throughput sequential reading, chunking, and memory buffer alignment
of genomic sequences into compact NumPy uint8 arrays (numerical encoding or ASCII)
and byte/string buffers suitable for CPU/GPU streaming pipelines.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Literal, Optional, Tuple, Union
import numpy as np
from Bio import SeqIO

# Numerical encoding translation lookup table: A=1, C=2, G=3, T=4, N=0
# Default all non-ACGT characters to 0 (N)
_ENCODE_LOOKUP = np.zeros(256, dtype=np.uint8)
_ENCODE_LOOKUP[ord("A")] = 1
_ENCODE_LOOKUP[ord("a")] = 1
_ENCODE_LOOKUP[ord("C")] = 2
_ENCODE_LOOKUP[ord("c")] = 2
_ENCODE_LOOKUP[ord("G")] = 3
_ENCODE_LOOKUP[ord("g")] = 3
_ENCODE_LOOKUP[ord("T")] = 4
_ENCODE_LOOKUP[ord("t")] = 4
_ENCODE_LOOKUP[ord("U")] = 4  # RNA U treated as T for CRISPR guide alignment
_ENCODE_LOOKUP[ord("u")] = 4


def encode_sequence_numpy(
    seq: Union[str, bytes, bytearray],
    mode: Literal["numeric", "ascii"] = "numeric",
) -> np.ndarray:
    """Convert nucleotide sequence into compact numpy uint8 array.

    Args:
        seq: Nucleotide sequence string, bytes, or bytearray.
        mode: Encoding mode:
            - "numeric": A=1, C=2, G=3, T=4, N=0 (all other characters map to 0).
            - "ascii": Raw ASCII uint8 representation (e.g. ord('A')=65).

    Returns:
        1D NumPy array with dtype np.uint8.
    """
    if isinstance(seq, str):
        raw_bytes = np.frombuffer(seq.encode("ascii", errors="replace"), dtype=np.uint8)
    elif isinstance(seq, (bytes, bytearray)):
        raw_bytes = np.frombuffer(seq, dtype=np.uint8)
    else:
        raise TypeError(f"Unsupported sequence type: {type(seq)}")

    if mode == "ascii":
        return raw_bytes.copy()
    elif mode == "numeric":
        return _ENCODE_LOOKUP[raw_bytes]
    else:
        raise ValueError(f"Unknown encoding mode: {mode}. Must be 'numeric' or 'ascii'.")


def decode_sequence_numpy(encoded: np.ndarray, mode: Literal["numeric", "ascii"] = "numeric") -> str:
    """Decode a numpy uint8 array back to an uppercase DNA string.

    Args:
        encoded: 1D NumPy uint8 array.
        mode: Encoding mode previously used ('numeric' or 'ascii').

    Returns:
        Decoded string.
    """
    if mode == "ascii":
        return bytes(encoded).decode("ascii")
    elif mode == "numeric":
        inv_map = {1: "A", 2: "C", 3: "G", 4: "T", 0: "N"}
        return "".join(inv_map.get(int(v), "N") for v in encoded)
    else:
        raise ValueError(f"Unknown encoding mode: {mode}")


class GenomicParser:
    """Parser for genomic FASTA files supporting memory-efficient streaming and chunking."""

    def __init__(
        self,
        filepath: Union[str, Path],
        chunk_size: Optional[int] = None,
        chunk_size_mb: Optional[float] = None,
    ) -> None:
        """Initialize the GenomicParser.

        Args:
            filepath: Path to the FASTA file.
            chunk_size: Size of output sequence chunks in bytes/characters.
                        (Defaults to 1MB if neither chunk_size nor chunk_size_mb is set).
            chunk_size_mb: Alternative specification of chunk size in Megabytes (1 MB = 1,048,576 bytes).

        Raises:
            FileNotFoundError: If the specified file does not exist.
            ValueError: If chunk size is less than 1.
        """
        self.filepath = Path(filepath)
        if not self.filepath.is_file():
            raise FileNotFoundError(f"FASTA file not found: {self.filepath.resolve()}")

        if chunk_size_mb is not None:
            self.chunk_size = int(chunk_size_mb * 1024 * 1024)
        elif chunk_size is not None:
            self.chunk_size = chunk_size
        else:
            self.chunk_size = 1_048_576  # Default 1MB

        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be a positive integer, got {self.chunk_size}")

    def stream_chunks(
        self,
        chunk_size_mb: Optional[float] = None,
        chunk_size: Optional[int] = None,
        overlap: int = 50,
        return_format: Literal["bytearray", "numpy", "numpy_ascii", "str"] = "bytearray",
    ) -> Generator[Tuple[str, int, Union[bytearray, np.ndarray, str]], None, None]:
        """Stream sequence chunks with overlap across records in the FASTA file.

        Splits genomic records across lines and boundaries while preserving exact sequence
        and overlap so that CRISPR target sites spanning window borders are not lost.

        Args:
            chunk_size_mb: Size of output chunk in Megabytes (overrides instance default if specified).
            chunk_size: Size of output chunk in bytes (overrides chunk_size_mb if specified).
            overlap: Number of overlapping bases preserved between consecutive chunks (default: 50).
                     Must satisfy 0 <= overlap < effective_chunk_size.
            return_format: Format of returned buffer:
                - "bytearray": Standard mutable contiguous byte buffer (ASCII).
                - "numpy": Compact NumPy uint8 array numerically encoded (A=1, C=2, G=3, T=4, N=0).
                - "numpy_ascii": NumPy uint8 array preserving ASCII byte values.
                - "str": Upper-case Python string.

        Yields:
            Tuple of:
                - record_id (str): Identifier of the genomic record/chromosome.
                - chunk_start_offset (int): Zero-based global character offset of this chunk.
                - buffer (bytearray | np.ndarray | str): Sequence chunk in requested format.

        Raises:
            ValueError: If overlap is invalid or return_format is unrecognized.
        """
        # Determine effective chunk size
        if chunk_size is not None:
            effective_chunk_size = chunk_size
        elif chunk_size_mb is not None:
            effective_chunk_size = int(chunk_size_mb * 1024 * 1024)
        else:
            effective_chunk_size = self.chunk_size

        if effective_chunk_size < 1:
            raise ValueError(f"effective_chunk_size must be positive, got {effective_chunk_size}")

        if overlap < 0 or overlap >= effective_chunk_size:
            raise ValueError(
                f"overlap must satisfy 0 <= overlap < chunk_size ({effective_chunk_size}), got {overlap}"
            )

        stride = effective_chunk_size - overlap

        with open(self.filepath, "r", encoding="utf-8") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq_str = str(record.seq).upper()
                seq_len = len(seq_str)

                if seq_len == 0:
                    continue

                for offset in range(0, seq_len, stride):
                    chunk_slice = seq_str[offset : offset + effective_chunk_size]

                    if return_format == "bytearray":
                        buf = bytearray(chunk_slice, encoding="ascii")
                    elif return_format == "numpy":
                        buf = encode_sequence_numpy(chunk_slice, mode="numeric")
                    elif return_format == "numpy_ascii":
                        buf = encode_sequence_numpy(chunk_slice, mode="ascii")
                    elif return_format == "str":
                        buf = chunk_slice
                    else:
                        raise ValueError(
                            f"Unknown return_format: '{return_format}'. "
                            f"Must be 'bytearray', 'numpy', 'numpy_ascii', or 'str'."
                        )

                    yield (record.id, offset, buf)

    def get_full_sequence_bytes(self) -> Generator[Tuple[str, bytearray], None, None]:
        """Yield entire sequence records as bytearrays without chunking.

        Yields:
            Tuple of (record_id, bytearray).
        """
        with open(self.filepath, "r", encoding="utf-8") as handle:
            for record in SeqIO.parse(handle, "fasta"):
                seq_str = str(record.seq).upper()
                yield (record.id, bytearray(seq_str, encoding="ascii"))
