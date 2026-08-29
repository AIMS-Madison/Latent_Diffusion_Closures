"""Random fields used by the controlled Missing Physics simulation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def q_wiener_spectrum(
    time_step: float,
    grid_shape: Sequence[int],
    domain: Sequence[float],
    alpha: float,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the square-root spectrum for a spatially correlated Q-Wiener field."""
    if time_step <= 0:
        raise ValueError("time_step must be positive")
    if len(grid_shape) != 2 or len(domain) != 2:
        raise ValueError("grid_shape and domain must each contain two entries")
    if any(size <= 0 for size in grid_shape) or any(length <= 0 for length in domain):
        raise ValueError("grid sizes and domain lengths must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")

    frequencies = [
        2.0
        * math.pi
        * torch.fft.fftfreq(size, d=length / size, device=device, dtype=dtype)
        for size, length in zip(grid_shape, domain)
    ]
    wave_x, wave_y = torch.meshgrid(*frequencies, indexing="ij")
    root_covariance = torch.exp(-0.5 * alpha * (wave_x.square() + wave_y.square()))
    normalization = math.prod(grid_shape) / math.sqrt(
        math.prod(domain) * time_step
    )
    return root_covariance * normalization


def sample_q_wiener_derivative(
    spectrum: torch.Tensor,
    batch_size: int,
    *,
    kappa: int = 1,
) -> torch.Tensor:
    """Sample a real-valued approximation of a Q-Wiener time derivative."""
    if spectrum.ndim != 2:
        raise ValueError("spectrum must be a two-dimensional tensor")
    if batch_size <= 0 or kappa <= 0:
        raise ValueError("batch_size and kappa must be positive")

    coefficients = torch.randn(
        kappa,
        batch_size,
        *spectrum.shape,
        2,
        device=spectrum.device,
        dtype=spectrum.dtype,
    ).sum(dim=0)
    complex_coefficients = torch.view_as_complex(coefficients)
    return torch.fft.ifft2(spectrum * complex_coefficients, dim=(-2, -1)).real


class GaussianRandomField:
    """Periodic Gaussian random field with Matérn-like spectral covariance."""

    def __init__(
        self,
        dim: int,
        size: int,
        *,
        alpha: float = 2.0,
        tau: float = 3.0,
        sigma: float | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if dim not in (1, 2, 3):
            raise ValueError("dim must be 1, 2, or 3")
        if size <= 0 or alpha <= 0 or tau <= 0:
            raise ValueError("size, alpha, and tau must be positive")

        self.dim = dim
        self.size = (size,) * dim
        self.device = torch.device(device or "cpu")
        self.dtype = dtype
        if sigma is None:
            sigma = tau ** (0.5 * (2.0 * alpha - dim))
        if sigma <= 0:
            raise ValueError("sigma must be positive")

        frequency = torch.fft.fftfreq(
            size,
            d=1.0 / size,
            device=self.device,
            dtype=dtype,
        )
        wave_numbers = torch.meshgrid(*([frequency] * dim), indexing="ij")
        squared_norm = sum(component.square() for component in wave_numbers)
        self.sqrt_eigenvalues = (
            size**dim
            * math.sqrt(2.0)
            * sigma
            * (4.0 * math.pi**2 * squared_norm + tau**2).pow(-alpha / 2.0)
        )
        self.sqrt_eigenvalues[(0,) * dim] = 0.0

    def sample(self, batch_size: int) -> torch.Tensor:
        """Draw ``batch_size`` fields with shape ``(batch, *self.size)``."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        coefficients = torch.randn(
            batch_size,
            *self.size,
            2,
            device=self.device,
            dtype=self.dtype,
        )
        coefficients = torch.view_as_complex(coefficients)
        coefficients = coefficients * self.sqrt_eigenvalues
        dimensions = tuple(range(1, self.dim + 1))
        return torch.fft.ifftn(coefficients, dim=dimensions).real


GaussianRF = GaussianRandomField


__all__ = [
    "GaussianRF",
    "GaussianRandomField",
    "q_wiener_spectrum",
    "sample_q_wiener_derivative",
]
