"""Unit tests for GPU hardware discovery and launch configuration calculations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from geneweaver.gpu_info import (
    CUDAEnvironmentReport,
    GPUDeviceInfo,
    GridLaunchConfig,
    inspect_cuda_environment,
    recommend_grid_config,
)


class TestGPUInfoAndLaunchConfig:
    """Test suite for Day 6 GPU discovery, capability checks, and grid calculations."""

    def test_recommend_grid_config_basic_math(self):
        """Test grid launch calculation logic: blocks, threads, warps, and shared memory."""
        seq_len = 1024  # 1024 bases
        threads_per_block = 256
        guide_len = 20

        # num_windows = 1024 - 20 + 1 = 1005
        # blocks = ceil(1005 / 256) = 4
        cfg = recommend_grid_config(
            sequence_length=seq_len,
            threads_per_block=threads_per_block,
            guide_length=guide_len,
            bytes_per_element=1,
        )

        assert cfg.sequence_length == 1024
        assert cfg.threads_per_block == 256
        assert cfg.blocks_per_grid == 4
        assert cfg.total_threads == 4 * 256
        assert cfg.warp_count_per_block == 8  # 256 / 32
        assert cfg.global_memory_bytes == 1024
        # shared_mem = 20 + (256 + 20 - 1) = 20 + 275 = 295 bytes
        assert cfg.shared_memory_bytes_per_block == 295

    def test_recommend_grid_config_one_megabase_chunk(self):
        """Test grid launch calculation for a standard 1MB (1,048,576 bytes) chunk."""
        one_mb = 1024 * 1024
        cfg = recommend_grid_config(sequence_length=one_mb, threads_per_block=256)

        assert cfg.global_memory_mb == 1.0
        assert cfg.blocks_per_grid > 4000
        assert cfg.blocks_per_grid * 256 >= (one_mb - 20 + 1)
        assert cfg.shared_memory_bytes_per_block < 49152  # Must be well within 48KB limit

    def test_recommend_grid_config_boundary_and_invalid_inputs(self):
        """Test boundary inputs and exceptions for non-positive dimensions."""
        # Non-positive sequence length
        with pytest.raises(ValueError, match="sequence_length must be a positive integer"):
            recommend_grid_config(sequence_length=0)

        with pytest.raises(ValueError, match="sequence_length must be a positive integer"):
            recommend_grid_config(sequence_length=-100)

        # Non-positive threads per block
        with pytest.raises(ValueError, match="threads_per_block must be a positive integer"):
            recommend_grid_config(sequence_length=1000, threads_per_block=0)

        # Sequence shorter than guide length
        cfg_short = recommend_grid_config(sequence_length=10, guide_length=20)
        assert cfg_short.blocks_per_grid == 1
        assert cfg_short.total_threads == 256

    def test_inspect_cuda_environment_mocked_gpu_present(self):
        """Test inspect_cuda_environment when a GPU is successfully detected."""
        fake_gpu = MagicMock()
        fake_gpu.name = b"NVIDIA RTX 4090"
        fake_gpu.compute_capability = (8, 9)
        fake_gpu.MULTIPROCESSOR_COUNT = 128
        fake_gpu.MAX_THREADS_PER_BLOCK = 1024
        fake_gpu.WARP_SIZE = 32
        fake_gpu.PCI_BUS_ID = 1
        fake_gpu.PCI_DEVICE_ID = 0
        fake_gpu.__enter__ = MagicMock(return_value=fake_gpu)
        fake_gpu.__exit__ = MagicMock(return_value=None)

        with patch("numba.cuda.gpus", [fake_gpu]), \
             patch("geneweaver.gpu_info._query_pynvml_memory", return_value=(24576.0, 20480.0, "550.54")), \
             patch("numba.cuda.cudadrv.nvvm.NVVM") as mock_nvvm:
            mock_nvvm.return_value.data_layout = "layout"

            report = inspect_cuda_environment()
            assert report.cuda_available is True
            assert len(report.devices) == 1
            dev = report.devices[0]
            assert dev.name == "NVIDIA RTX 4090"
            assert dev.compute_capability == (8, 9)
            assert dev.compute_capability_str == "8.9"
            assert dev.total_memory_mb == 24576.0
            assert dev.free_memory_mb == 20480.0
            assert dev.total_memory_gb == 24.0
            assert dev.multiprocessor_count == 128
            assert report.kernel_compilation_ready is True

    def test_inspect_cuda_environment_mocked_no_gpu_fallback(self):
        """Test inspect_cuda_environment graceful fallback when no GPU is present."""
        with patch("numba.cuda.gpus", []):
            report = inspect_cuda_environment()
            assert report.cuda_available is False
            assert len(report.devices) == 0
            assert report.kernel_compilation_ready is False
            assert "No CUDA-capable GPU detected" in report.diagnostic_message
