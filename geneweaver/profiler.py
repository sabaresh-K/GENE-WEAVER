"""CPU profiling utilities using cProfile and pstats for GeneWeaver.

Pinpoints exact functions, loops, and memory buffer operations consuming
execution cycles during sliding-window sequence alignment.
"""

from __future__ import annotations

import cProfile
import io
import pstats
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

from geneweaver.aligner_cpu import CPUAligner
from geneweaver.parser import encode_sequence_numpy


@dataclass
class ProfilerHotspotEntry:
    """Represents a profiled function hotspot entry."""

    function_name: str
    filename_line: str
    ncalls: int
    tottime: float
    percall_tottime: float
    cumtime: float
    percall_cumtime: float


@dataclass
class ProfilingReport:
    """Summary container for cProfile analysis."""

    total_time: float
    total_calls: int
    top_cumulative: List[ProfilerHotspotEntry]
    top_internal: List[ProfilerHotspotEntry]
    formatted_pstats_table: str


def profile_alignment_hotspots(
    sequence_data: Union[str, bytes, bytearray, np.ndarray],
    guide_rna: str = "GTCACCGTAGTTCATAGCGG",
    max_mismatches: int = 3,
    top_n: int = 15,
) -> ProfilingReport:
    """Profile sequence alignment execution using cProfile and extract key bottlenecks.

    Args:
        sequence_data: Sequence chunk to align against (str, bytes, or np.ndarray).
        guide_rna: Query guide RNA string.
        max_mismatches: Max Hamming distance threshold.
        top_n: Number of top hotspot entries to include in report.

    Returns:
        ProfilingReport containing structured hotspot records and formatted summary.
    """
    if not isinstance(sequence_data, np.ndarray):
        chunk_uint8 = encode_sequence_numpy(sequence_data, mode="numeric")
    else:
        chunk_uint8 = sequence_data

    aligner = CPUAligner(guide_rna=guide_rna, max_mismatches=max_mismatches)

    profiler = cProfile.Profile()
    profiler.enable()

    # Target workload
    aligner.align_chunk(chunk_uint8)

    profiler.disable()

    # Format pstats table
    stream = io.StringIO()
    ps = pstats.Stats(profiler, stream=stream)
    ps.strip_dirs()
    ps.sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(top_n)
    formatted_cum_table = stream.getvalue()

    # Extract structured entries
    stats_dict = ps.stats

    def _convert_entry(func_tuple: Tuple[str, int, str], metrics: Tuple[Any, ...]) -> ProfilerHotspotEntry:
        filename, line, func_name = func_tuple
        cc, nc, tottime, cumtime, _ = metrics
        percall_tot = tottime / nc if nc > 0 else 0.0
        percall_cum = cumtime / cc if cc > 0 else 0.0
        return ProfilerHotspotEntry(
            function_name=func_name,
            filename_line=f"{filename}:{line}",
            ncalls=nc,
            tottime=round(tottime, 5),
            percall_tottime=round(percall_tot, 6),
            cumtime=round(cumtime, 5),
            percall_cumtime=round(percall_cum, 6),
        )

    # Sort by cumulative time
    sorted_by_cum = sorted(stats_dict.items(), key=lambda kv: kv[1][3], reverse=True)[:top_n]
    top_cum_entries = [_convert_entry(k, v) for k, v in sorted_by_cum]

    # Sort by internal total time
    sorted_by_tot = sorted(stats_dict.items(), key=lambda kv: kv[1][2], reverse=True)[:top_n]
    top_tot_entries = [_convert_entry(k, v) for k, v in sorted_by_tot]

    return ProfilingReport(
        total_time=round(ps.total_tt, 5),
        total_calls=ps.total_calls,
        top_cumulative=top_cum_entries,
        top_internal=top_tot_entries,
        formatted_pstats_table=formatted_cum_table,
    )
