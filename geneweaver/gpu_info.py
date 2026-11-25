"""NVIDIA GPU discovery, hardware inspection, and CUDA launch configuration for GeneWeaver.

Provides hardware capability inspection, VRAM queries via pynvml/nvidia-ml-py and Numba CUDA,
graceful error diagnostic reporting across CUDA-enabled and CPU-only environments,
and mathematical launch grid dimension calculations for sliding-window sequence alignment.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GPUDeviceInfo:
    """Hardware architecture and memory properties for a detected NVIDIA GPU."""

    device_id: int
    name: str
    compute_capability: Tuple[int, int]
    compute_capability_str: str
    total_memory_mb: float
    total_memory_gb: float
    free_memory_mb: Optional[float]
    multiprocessor_count: int
    max_threads_per_block: int
    warp_size: int
    clock_rate_khz: Optional[int] = None
    pci_bus_id: Optional[int] = None
    pci_device_id: Optional[int] = None


@dataclass
class GridLaunchConfig:
    """Calculated 1D kernel execution grid configuration for sequence chunks."""

    sequence_length: int
    threads_per_block: int
    blocks_per_grid: int
    total_threads: int
    global_memory_bytes: int
    global_memory_mb: float
    shared_memory_bytes_per_block: int
    warp_count_per_block: int
    hardware_occupancy_ratio: Optional[float] = None


@dataclass
class CUDAEnvironmentReport:
    """Environment diagnostics and detected GPU inventory."""

    cuda_available: bool
    driver_version: Optional[str]
    devices: List[GPUDeviceInfo]
    diagnostic_message: str
    kernel_compilation_ready: bool


def _query_pynvml_memory(device_idx: int) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    """Query VRAM details and driver version via nvidia-ml-py or pynvml."""
    try:
        import pynvml
        pynvml.nvmlInit()
        driver_ver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver_ver, bytes):
            driver_ver = driver_ver.decode("utf-8")
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_mb = mem_info.total / (1024 * 1024)
        free_mb = mem_info.free / (1024 * 1024)
        pynvml.nvmlShutdown()
        return round(total_mb, 2), round(free_mb, 2), driver_ver
    except Exception:
        return None, None, None


def inspect_cuda_environment() -> CUDAEnvironmentReport:
    """Query and inspect NVIDIA GPU hardware, CUDA runtime, and driver status.

    Returns:
        CUDAEnvironmentReport containing hardware specifications and diagnostic details.
    """
    devices: List[GPUDeviceInfo] = []
    cuda_available = False
    kernel_compilation_ready = False
    driver_version: Optional[str] = None
    diagnostics: List[str] = []

    try:
        from numba import cuda
    except ImportError as e:
        return CUDAEnvironmentReport(
            cuda_available=False,
            driver_version=None,
            devices=[],
            diagnostic_message=f"Numba package not found or corrupted: {e}",
            kernel_compilation_ready=False,
        )

    # 1. Query Driver / Hardware Level
    try:
        cuda_device_count = len(cuda.gpus)
    except Exception as e:
        cuda_device_count = 0
        diagnostics.append(f"CUDA driver error: {e}")

    if cuda_device_count == 0:
        diagnostics.append(
            "No CUDA-capable GPU detected or NVIDIA display driver is missing/inaccessible."
        )
        return CUDAEnvironmentReport(
            cuda_available=False,
            driver_version=None,
            devices=[],
            diagnostic_message="; ".join(diagnostics),
            kernel_compilation_ready=False,
        )

    cuda_available = True

    # 2. Inspect Hardware Properties for Each Detected GPU
    for idx, dev in enumerate(cuda.gpus):
        with dev:
            raw_name = dev.name
            dev_name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
            cc = dev.compute_capability
            cc_str = f"{cc[0]}.{cc[1]}"

            # Attempt VRAM query
            total_mb, free_mb, drv = _query_pynvml_memory(idx)
            if drv and not driver_version:
                driver_version = drv

            # Fallback estimation if NVML is unavailable
            if total_mb is None:
                total_mb = 4096.0  # Conservative estimate
                free_mb = None

            total_gb = round(total_mb / 1024.0, 2)

            gpu_info = GPUDeviceInfo(
                device_id=idx,
                name=dev_name,
                compute_capability=cc,
                compute_capability_str=cc_str,
                total_memory_mb=total_mb,
                total_memory_gb=total_gb,
                free_memory_mb=free_mb,
                multiprocessor_count=getattr(dev, "MULTIPROCESSOR_COUNT", 0),
                max_threads_per_block=getattr(dev, "MAX_THREADS_PER_BLOCK", 1024),
                warp_size=getattr(dev, "WARP_SIZE", 32),
                pci_bus_id=getattr(dev, "PCI_BUS_ID", None),
                pci_device_id=getattr(dev, "PCI_DEVICE_ID", None),
            )
            devices.append(gpu_info)

    # 3. Check NVVM Kernel Compilation Readiness
    try:
        from numba.cuda.cudadrv.nvvm import NVVM
        _ = NVVM().data_layout
        kernel_compilation_ready = True
        diagnostics.append("NVIDIA GPU detected and CUDA NVVM compiler toolchain is active.")
    except Exception as e:
        kernel_compilation_ready = False
        diagnostics.append(
            f"GPU hardware detected ({devices[0].name}), but CUDA Toolkit compiler (nvvm.dll / libNVVM) "
            f"is missing from PATH/environment: {e}. "
            "Install CUDA Toolkit or conda cudatoolkit for @cuda.jit runtime compilation."
        )

    return CUDAEnvironmentReport(
        cuda_available=cuda_available,
        driver_version=driver_version,
        devices=devices,
        diagnostic_message=" | ".join(diagnostics),
        kernel_compilation_ready=kernel_compilation_ready,
    )


def recommend_grid_config(
    sequence_length: int,
    threads_per_block: int = 256,
    guide_length: int = 20,
    bytes_per_element: int = 1,
    max_shared_mem_per_block: int = 49152,
) -> GridLaunchConfig:
    """Calculate 1D grid and block dimensions for sequence chunk alignment.

    Computes:
    - `blocks_per_grid = ceil(num_windows / threads_per_block)`
    - Memory footprint for input chunk and guide buffer.
    - Safe shared memory allocation per block (e.g. guide cache + chunk halo).

    Args:
        sequence_length: Size of the genomic chunk in base pairs (bytes).
        threads_per_block: Number of CUDA threads per thread block (default: 256, typically multiple of 32).
        guide_length: Size of the CRISPR target query in bp (default: 20).
        bytes_per_element: Storage size per nucleotide (1 for uint8).
        max_shared_mem_per_block: Hardware shared memory limit in bytes (default 48KB).

    Returns:
        GridLaunchConfig with grid, block, thread, and memory estimations.

    Raises:
        ValueError: If sequence_length or threads_per_block are non-positive.
    """
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be a positive integer, got {sequence_length}")
    if threads_per_block <= 0:
        raise ValueError(f"threads_per_block must be a positive integer, got {threads_per_block}")

    # Number of candidate alignment windows
    num_windows = max(0, sequence_length - guide_length + 1)
    if num_windows == 0:
        blocks_per_grid = 1
    else:
        blocks_per_grid = math.ceil(num_windows / threads_per_block)

    total_threads = blocks_per_grid * threads_per_block
    global_memory_bytes = sequence_length * bytes_per_element
    global_memory_mb = round(global_memory_bytes / (1024 * 1024), 3)

    # Shared memory estimation:
    # Guide RNA buffer (guide_length bytes) + per-block chunk slice halo (threads_per_block + guide_length - 1)
    shared_mem_needed = (guide_length * bytes_per_element) + (
        (threads_per_block + guide_length - 1) * bytes_per_element
    )
    shared_mem_safe = min(shared_mem_needed, max_shared_mem_per_block)
    warps = math.ceil(threads_per_block / 32)

    return GridLaunchConfig(
        sequence_length=sequence_length,
        threads_per_block=threads_per_block,
        blocks_per_grid=blocks_per_grid,
        total_threads=total_threads,
        global_memory_bytes=global_memory_bytes,
        global_memory_mb=global_memory_mb,
        shared_memory_bytes_per_block=shared_mem_safe,
        warp_count_per_block=warps,
    )
