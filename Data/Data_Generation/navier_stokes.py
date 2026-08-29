from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from tqdm.auto import trange

from .random_forcing import q_wiener_spectrum, sample_q_wiener_derivative


@dataclass(frozen=True)
class MissingPhysicsBatch:
    """Recorded fields from one simulation batch."""

    vorticity: torch.Tensor
    nonlinear: torch.Tensor
    diffusion: torch.Tensor
    times: torch.Tensor


def _validated_step_count(duration: float, time_step: float, name: str) -> int:
    steps = round(duration / time_step)
    if not math.isclose(steps * time_step, duration, rel_tol=0.0, abs_tol=1e-10):
        raise ValueError(f"{name} must be an integer multiple of time_step")
    return steps


@torch.no_grad()
def simulate_navier_stokes_2d(
    domain: tuple[float, float],
    initial_vorticity: torch.Tensor,
    deterministic_forcing: torch.Tensor,
    *,
    viscosity: float,
    final_time: float,
    time_step: float,
    burn_in_time: float = 0.0,
    record_interval_steps: int = 1,
    stochastic_forcing: dict[str, float | int] | None = None,
    output_stride: int = 1,
    show_progress: bool = False,
) -> MissingPhysicsBatch:
    """Simulate vorticity and record the known and missing right-hand-side terms."""
    if initial_vorticity.ndim != 3:
        raise ValueError("initial_vorticity must have shape (batch, nx, ny)")
    if initial_vorticity.shape[-2] != initial_vorticity.shape[-1]:
        raise ValueError("only square spatial grids are supported")
    if len(domain) != 2 or any(length <= 0 for length in domain):
        raise ValueError("domain must contain two positive lengths")
    if viscosity <= 0 or time_step <= 0 or final_time <= 0:
        raise ValueError("viscosity, time_step, and final_time must be positive")
    if not 0 <= burn_in_time < final_time:
        raise ValueError("burn_in_time must be in [0, final_time)")
    if record_interval_steps <= 0 or output_stride <= 0:
        raise ValueError("record_interval_steps and output_stride must be positive")

    batch_size, grid_x, grid_y = initial_vorticity.shape
    if grid_x % output_stride or grid_y % output_stride:
        raise ValueError("output_stride must divide both spatial dimensions")
    if deterministic_forcing.shape not in {
        (grid_x, grid_y),
        (batch_size, grid_x, grid_y),
    }:
        raise ValueError("deterministic_forcing has an incompatible shape")

    total_steps = _validated_step_count(final_time, time_step, "final_time")
    burn_in_step = _validated_step_count(burn_in_time, time_step, "burn_in_time")
    record_steps = tuple(range(burn_in_step, total_steps, record_interval_steps))
    if not record_steps:
        raise ValueError("the requested time range does not contain a snapshot")
    record_lookup = {step: index for index, step in enumerate(record_steps)}

    device = initial_vorticity.device
    dtype = initial_vorticity.dtype
    frequency_x = 2.0 * math.pi * torch.fft.fftfreq(
        grid_x,
        d=domain[0] / grid_x,
        device=device,
        dtype=dtype,
    )
    frequency_y = 2.0 * math.pi * torch.fft.rfftfreq(
        grid_y,
        d=domain[1] / grid_y,
        device=device,
        dtype=dtype,
    )
    wave_x, wave_y = torch.meshgrid(frequency_x, frequency_y, indexing="ij")
    laplacian = wave_x.square() + wave_y.square()
    inverse_laplacian = laplacian.clone()
    inverse_laplacian[0, 0] = 1.0

    integer_x = torch.fft.fftfreq(grid_x, d=1.0 / grid_x, device=device)
    integer_y = torch.fft.rfftfreq(grid_y, d=1.0 / grid_y, device=device)
    mode_x, mode_y = torch.meshgrid(integer_x, integer_y, indexing="ij")
    cutoff = min(grid_x, grid_y) / 3.0
    dealias = ((mode_x.abs() <= cutoff) & (mode_y.abs() <= cutoff)).to(dtype)

    vorticity_fft = torch.fft.rfft2(initial_vorticity)
    forcing_fft = torch.fft.rfft2(deterministic_forcing)
    if forcing_fft.ndim == 2:
        forcing_fft = forcing_fft.unsqueeze(0)

    noise_spectrum = None
    noise_scale = 0.0
    noise_kappa = 1
    if stochastic_forcing is not None:
        noise_spectrum = q_wiener_spectrum(
            time_step,
            (grid_x, grid_y),
            domain,
            float(stochastic_forcing["alpha"]),
            device=device,
            dtype=dtype,
        )
        noise_scale = float(stochastic_forcing["sigma"])
        noise_kappa = int(stochastic_forcing["kappa"])

    output_shape = (
        batch_size,
        grid_x // output_stride,
        grid_y // output_stride,
        len(record_steps),
    )
    vorticity_output = torch.empty(output_shape, device=device, dtype=dtype)
    nonlinear_output = torch.empty_like(vorticity_output)
    diffusion_output = torch.empty_like(vorticity_output)

    iterator = trange(total_steps, disable=not show_progress, desc="Simulating")
    for step in iterator:
        stream_function_fft = vorticity_fft / inverse_laplacian
        velocity_x = torch.fft.irfft2(
            1j * wave_y * stream_function_fft,
            s=(grid_x, grid_y),
        )
        velocity_y = torch.fft.irfft2(
            -1j * wave_x * stream_function_fft,
            s=(grid_x, grid_y),
        )
        gradient_x = torch.fft.irfft2(
            1j * wave_x * vorticity_fft,
            s=(grid_x, grid_y),
        )
        gradient_y = torch.fft.irfft2(
            1j * wave_y * vorticity_fft,
            s=(grid_x, grid_y),
        )
        advection_fft = torch.fft.rfft2(
            velocity_x * gradient_x + velocity_y * gradient_y
        )
        advection_fft = advection_fft * dealias

        if noise_spectrum is None:
            stochastic_fft = torch.zeros_like(vorticity_fft)
        else:
            noise = sample_q_wiener_derivative(
                noise_spectrum,
                batch_size,
                kappa=noise_kappa,
            )
            stochastic_fft = noise_scale * torch.fft.rfft2(noise)

        nonlinear_fft = -advection_fft + stochastic_fft
        diffusion_fft = -viscosity * laplacian * vorticity_fft + forcing_fft

        if step in record_lookup:
            output_index = record_lookup[step]
            spatial_slice = (..., slice(None, None, output_stride), slice(None, None, output_stride))
            vorticity = torch.fft.irfft2(
                vorticity_fft,
                s=(grid_x, grid_y),
            )
            nonlinear = torch.fft.irfft2(
                nonlinear_fft,
                s=(grid_x, grid_y),
            )
            diffusion = torch.fft.irfft2(
                diffusion_fft,
                s=(grid_x, grid_y),
            )
            vorticity_output[..., output_index] = vorticity[spatial_slice]
            nonlinear_output[..., output_index] = nonlinear[spatial_slice]
            diffusion_output[..., output_index] = diffusion[spatial_slice]

        vorticity_fft = (
            vorticity_fft
            + time_step * (nonlinear_fft + forcing_fft)
            - 0.5 * time_step * viscosity * laplacian * vorticity_fft
        ) / (1.0 + 0.5 * time_step * viscosity * laplacian)
        if not torch.isfinite(vorticity_fft).all():
            raise FloatingPointError(f"simulation became non-finite at step {step}")

    times = torch.tensor(record_steps, device=device, dtype=dtype) * time_step
    return MissingPhysicsBatch(
        vorticity=vorticity_output,
        nonlinear=nonlinear_output,
        diffusion=diffusion_output,
        times=times,
    )


__all__ = ["MissingPhysicsBatch", "simulate_navier_stokes_2d"]
