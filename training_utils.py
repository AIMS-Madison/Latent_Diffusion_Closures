"""Small runtime helpers shared by training and evaluation scripts."""

from __future__ import annotations

import numpy as np
import torch


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return CUDA when available, otherwise CPU."""
    if prefer_cuda and torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
        return torch.device("cuda")
    print("CUDA is not available. Using CPU.")
    return torch.device("cpu")


def to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move a tensor to the selected runtime device."""
    return tensor.to(device)


def safe_cuda_synchronize(device: torch.device | str | None = None) -> None:
    """Synchronize CUDA only when the selected device is CUDA."""
    if device is None:
        should_sync = torch.cuda.is_available()
    else:
        should_sync = torch.device(device).type == "cuda"

    if should_sync and torch.cuda.is_available():
        torch.cuda.synchronize()


def create_ticks_labels(
    size: int,
    step: int = 20,
    reference_size: int = 64,
) -> tuple[np.ndarray, list[str]]:
    """Create plot tick positions scaled from a reference grid."""
    ticks = np.arange(0, size, step * size / reference_size)
    tick_labels = [str(int(tick)) for tick in ticks]
    return ticks, tick_labels
