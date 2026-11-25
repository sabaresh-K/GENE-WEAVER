"""Day 6 NVIDIA GPU discovery, capability check, and CUDA runtime verification CLI.

Inspects system hardware, displays compute capabilities, VRAM, and launch grid
recommendations, and tests minimal kernel dispatch if NVVM is available.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from geneweaver.gpu_info import inspect_cuda_environment, recommend_grid_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for GPU environment inspection."""
    parser = argparse.ArgumentParser(
        description="GeneWeaver Day 6 NVIDIA GPU Discovery & CUDA Capability Inspection"
    )
    parser.add_argument(
        "--test-chunk-mb",
        type=float,
        default=1.0,
        help="Chunk size in MB to evaluate for grid/block recommendations (default: 1.0 MB).",
    )
    parser.add_argument(
        "--threads-per-block",
        type=int,
        default=256,
        help="Thread block size to evaluate (default: 256).",
    )
    return parser.parse_args()


def test_minimal_cuda_kernel() -> bool:
    """Attempt a minimal Numba CUDA kernel launch on a uint8 array."""
    try:
        from numba import cuda

        @cuda.jit
        def _add_one_kernel(arr):
            idx = cuda.grid(1)
            if idx < arr.size:
                arr[idx] += 1

        data = np.array([1, 2, 3, 4], dtype=np.uint8)
        d_data = cuda.to_device(data)
        _add_one_kernel[1, 32](d_data)
        result = d_data.copy_to_host()

        expected = np.array([2, 3, 4, 5], dtype=np.uint8)
        if np.array_equal(result, expected):
            print("[OK] Minimal CUDA kernel compiled, executed, and validated successfully!")
            return True
        else:
            print(f"[-] Kernel output mismatch: got {result}, expected {expected}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"[-] Minimal CUDA kernel execution failed: {e}", file=sys.stderr)
        return False


def main() -> int:
    """Execute Day 6 GPU environment check and report."""
    args = parse_args()

    print("=" * 80)
    print("      GeneWeaver: Day 6 NVIDIA GPU Discovery & Capability Check")
    print("=" * 80)

    report = inspect_cuda_environment()

    # 1. Driver & Hardware Overview
    print(f"[*] CUDA Hardware Detected : {'YES' if report.cuda_available else 'NO'}")
    if report.driver_version:
        print(f"[*] NVIDIA Driver Version   : {report.driver_version}")
    print(f"[*] Diagnostic Message     : {report.diagnostic_message}")
    print(f"[*] NVVM Compilation Ready : {'YES' if report.kernel_compilation_ready else 'NO'}")

    if not report.cuda_available:
        print("\n[!] WARNING: No CUDA-capable GPU detected.")
        print("    GeneWeaver will continue to support the pure-Python CPU engine (`CPUAligner`).")
        print("=" * 80)
        return 0

    # 2. Detailed Hardware Breakdown
    print("\n" + "-" * 80)
    print(f"  Detected GPU Device Inventory ({len(report.devices)} Found)")
    print("-" * 80)

    for dev in report.devices:
        print(f"  [Device {dev.device_id}] : {dev.name}")
        print(f"    - Compute Capability : SM {dev.compute_capability_str} (Major: {dev.compute_capability[0]}, Minor: {dev.compute_capability[1]})")
        print(f"    - Total Memory (VRAM): {dev.total_memory_mb:.1f} MB ({dev.total_memory_gb:.2f} GB)")
        if dev.free_memory_mb is not None:
            print(f"    - Free VRAM          : {dev.free_memory_mb:.1f} MB")
        print(f"    - Streaming MPs (SMs): {dev.multiprocessor_count}")
        print(f"    - Max Threads / Block: {dev.max_threads_per_block}")
        print(f"    - Warp Size          : {dev.warp_size} threads")
        if dev.pci_bus_id is not None:
            print(f"    - PCI Bus ID / Device: Bus {dev.pci_bus_id}, Device {dev.pci_device_id}")

    # 3. Recommended Grid / Block Launch Configuration for Genomic Chunks
    chunk_bytes = int(args.test_chunk_mb * 1024 * 1024)
    grid_cfg = recommend_grid_config(
        sequence_length=chunk_bytes,
        threads_per_block=args.threads_per_block,
        guide_length=20,
    )

    print("\n" + "-" * 80)
    print(f"  Recommended CUDA Launch Configuration for {args.test_chunk_mb:.2f} MB Chunk")
    print("-" * 80)
    print(f"  - Sequence Length (Bytes) : {grid_cfg.sequence_length:,} bp")
    print(f"  - Threads per Block       : {grid_cfg.threads_per_block} (multiple of warp 32)")
    print(f"  - Blocks per Grid         : {grid_cfg.blocks_per_grid:,} (1D Grid Dim)")
    print(f"  - Total Threads Launched  : {grid_cfg.total_threads:,}")
    print(f"  - Global Memory Required  : {grid_cfg.global_memory_mb:.2f} MB")
    print(f"  - Shared Memory / Block   : {grid_cfg.shared_memory_bytes_per_block} bytes (< 48 KB limit)")
    print(f"  - Warps per Block         : {grid_cfg.warp_count_per_block}")

    # 4. Kernel Compilation Check
    print("\n" + "-" * 80)
    print("  Testing CUDA Kernel Execution (uint8 Array)")
    print("-" * 80)

    if report.kernel_compilation_ready:
        success = test_minimal_cuda_kernel()
        if success:
            print("\n[OK] Hardware & CUDA compilation ready for Day 7 CUDA sliding-window kernels!")
        else:
            print("\n[!] Kernel launch encountered an issue.")
    else:
        print("[!] Note: NVVM compiler library (`nvvm.dll`) not linked in current Python environment.")
        print("    Hardware discovery and launch configurations were computed successfully.")
        print("    For runtime JIT compilation, ensure CUDA Toolkit / cudatoolkit is present on PATH.")

    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
