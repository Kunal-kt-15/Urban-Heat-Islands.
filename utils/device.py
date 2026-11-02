"""
Hardware and compute device management utilities.
Supports automatic CPU/CUDA selection and GPU telemetry.
"""

import logging
import torch

logger = logging.getLogger(__name__)


def get_device(preferred_device: str = None) -> torch.device:
    """
    Returns the appropriate torch.device based on availability or preference.
    """
    if preferred_device is not None:
        return torch.device(preferred_device)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        logger.info("Using GPU: %s (%.0f MB VRAM)", device_name, vram_mb)
        return device
    else:
        logger.info("CUDA not available. Using CPU.")
        return torch.device("cpu")


def get_device_info() -> dict:
    """
    Returns hardware diagnostics dictionary.
    """
    cuda_avail = torch.cuda.is_available()
    info = {
        "cuda_available": cuda_avail,
        "device_count": torch.cuda.device_count() if cuda_avail else 0,
        "current_device": torch.cuda.get_device_name(0) if cuda_avail else "CPU",
        "torch_version": torch.__version__
    }
    if cuda_avail:
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        alloc_mem = torch.cuda.memory_allocated(0) / (1024**2)
        info["vram_total_mb"] = round(total_mem, 1)
        info["vram_allocated_mb"] = round(alloc_mem, 1)
    return info
