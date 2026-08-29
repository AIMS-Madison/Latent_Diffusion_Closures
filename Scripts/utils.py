from __future__ import annotations

import os
import random

import numpy as np
import torch


def select_device(requested: str = "auto") -> torch.device:
    """Resolve an explicit device or select CUDA when available."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def synchronize(device: torch.device | str) -> None:
    if torch.device(device).type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def mean_squared_error(reference: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    return torch.mean((reference - estimate) ** 2)


def relative_frobenius_error(
    reference: torch.Tensor,
    estimate: torch.Tensor,
) -> torch.Tensor:
    difference = torch.linalg.vector_norm(
        (reference - estimate).flatten(start_dim=1),
        dim=1,
    )
    denominator = torch.linalg.vector_norm(reference.flatten(start_dim=1), dim=1)
    return torch.mean(difference / denominator.clamp_min(torch.finfo(reference.dtype).eps))


def get_sigmas_karras(
    steps: int,
    time_min: float,
    time_max: float,
    rho: float = 7.0,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Construct the nonuniform reverse-time schedule used for sampling."""
    if steps < 1:
        raise ValueError("steps must be positive")
    if not 0.0 <= time_min < time_max:
        raise ValueError("time bounds must satisfy 0 <= time_min < time_max")
    if rho <= 0:
        raise ValueError("rho must be positive")
    ramp = torch.linspace(0.0, 1.0, steps, device=device)
    min_inv_rho = time_min ** (1.0 / rho)
    max_inv_rho = time_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))


def moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size < 1:
        raise ValueError("window_size must be positive")
    return np.convolve(values, np.ones(window_size), "valid") / window_size


def energy_spectrum(
    fields: np.ndarray,
    lx: float = 1.0,
    ly: float = 1.0,
    smooth: bool = True,
    window_size: int = 5,
) -> dict[str, np.ndarray]:
    """Compute the isotropic spectrum of scalar 2-D field snapshots."""
    if fields.ndim != 3:
        raise ValueError("fields must have shape (samples, nx, ny)")
    _, nx, ny = fields.shape
    field_fft = np.fft.fftn(fields, axes=(1, 2)) / (nx * ny)
    spectral_energy = 0.5 * np.abs(field_fft) ** 2

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    wave_x, wave_y = np.meshgrid(kx, ky, indexing="ij")
    spacing = min(2.0 * np.pi / lx, 2.0 * np.pi / ly)
    shell_index = np.rint(np.sqrt(wave_x**2 + wave_y**2) / spacing).astype(int)

    spectrum = np.zeros(shell_index.max() + 1)
    for shell in range(spectrum.size):
        mask = shell_index == shell
        spectrum[shell] = np.mean(np.sum(spectral_energy[:, mask], axis=1))

    if smooth and spectrum.size >= window_size:
        smoothed = moving_average(spectrum, window_size)
        smoothed = np.pad(smoothed, (0, spectrum.size - smoothed.size))
        smoothed[: window_size - 1] = spectrum[: window_size - 1]
        spectrum = smoothed

    return {"k": spacing * np.arange(spectrum.size), "E": spectrum}
