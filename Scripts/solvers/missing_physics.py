from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import torch
from tqdm.auto import tqdm


ClosureSampler = Callable[[torch.Tensor], torch.Tensor]


@torch.no_grad()
def solve_vorticity_with_closure(
    domain: Sequence[float],
    initial_vorticity: torch.Tensor,
    forcing: torch.Tensor,
    viscosity: float,
    closure_sampler: ClosureSampler | None = None,
    *,
    time_step: float = 1e-3,
    integration_steps: int = 1,
    snapshot_interval: int = 1,
    model_evaluation_interval: int = 5,
    closure_noise_scale: float = 5e-5,
    show_progress: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Integrate vorticity while periodically sampling the missing nonlinear term."""
    if len(domain) != 2 or min(domain) <= 0:
        raise ValueError("domain must contain two positive lengths")
    if initial_vorticity.ndim != 3:
        raise ValueError("initial_vorticity must have shape (batch, nx, ny)")
    if initial_vorticity.shape[-2] != initial_vorticity.shape[-1]:
        raise ValueError("only square spatial grids are supported")
    if forcing.shape not in {
        initial_vorticity.shape[-2:],
        initial_vorticity.shape,
    }:
        raise ValueError("forcing has an incompatible shape")
    if viscosity <= 0 or time_step <= 0:
        raise ValueError("viscosity and time_step must be positive")
    if closure_noise_scale < 0:
        raise ValueError("closure_noise_scale cannot be negative")
    if integration_steps < 1 or snapshot_interval < 1:
        raise ValueError("integration_steps and snapshot_interval must be positive")
    if integration_steps % snapshot_interval != 0:
        raise ValueError("integration_steps must be divisible by snapshot_interval")
    if model_evaluation_interval < 1:
        raise ValueError("model_evaluation_interval must be positive")

    grid_size = initial_vorticity.shape[-1]
    vorticity_fft = torch.fft.rfft2(initial_vorticity)
    forcing_fft = torch.fft.rfft2(forcing)
    if forcing_fft.ndim < vorticity_fft.ndim:
        forcing_fft = forcing_fft.unsqueeze(0)

    frequency_x = torch.fft.fftfreq(
        grid_size,
        d=domain[0] / grid_size,
        device=initial_vorticity.device,
        dtype=initial_vorticity.dtype,
    )
    frequency_y = torch.fft.rfftfreq(
        grid_size,
        d=domain[1] / grid_size,
        device=initial_vorticity.device,
        dtype=initial_vorticity.dtype,
    )
    wave_x, wave_y = torch.meshgrid(frequency_x, frequency_y, indexing="ij")
    physical_wave_x = 2.0 * torch.pi * wave_x
    physical_wave_y = 2.0 * torch.pi * wave_y
    laplacian = physical_wave_x.square() + physical_wave_y.square()

    snapshot_count = integration_steps // snapshot_interval + 1
    snapshots = torch.zeros(
        *initial_vorticity.shape,
        snapshot_count,
        device=initial_vorticity.device,
        dtype=initial_vorticity.dtype,
    )
    snapshot_times = torch.zeros(
        snapshot_count,
        device=initial_vorticity.device,
        dtype=initial_vorticity.dtype,
    )
    snapshots[..., 0] = initial_vorticity
    current_closure: torch.Tensor | None = None
    start_time = time.perf_counter()

    progress = tqdm(
        range(integration_steps),
        desc="Vorticity integration",
        disable=not show_progress,
    )
    for step in progress:
        vorticity = torch.fft.irfft2(
            vorticity_fft,
            s=(grid_size, grid_size),
        )
        if closure_sampler is None:
            closure_fft: torch.Tensor | float = 0.0
        else:
            if step % model_evaluation_interval == 0 or current_closure is None:
                current_closure = closure_sampler(vorticity)
                if current_closure.shape != vorticity.shape:
                    raise ValueError("closure sampler returned an incompatible shape")
                current_closure = current_closure.to(
                    device=vorticity.device,
                    dtype=vorticity.dtype,
                )
            elif closure_noise_scale:
                current_closure = (
                    current_closure
                    + closure_noise_scale * torch.randn_like(current_closure)
                )
            closure_fft = torch.fft.rfft2(current_closure)

        vorticity_fft = (
            vorticity_fft
            + time_step * forcing_fft
            + time_step * closure_fft
            - 0.5 * time_step * viscosity * laplacian * vorticity_fft
        ) / (1.0 + 0.5 * time_step * viscosity * laplacian)
        if not torch.isfinite(vorticity_fft).all():
            raise FloatingPointError(f"integration became non-finite at step {step}")

        if (step + 1) % snapshot_interval == 0:
            snapshot_index = (step + 1) // snapshot_interval
            snapshots[..., snapshot_index] = torch.fft.irfft2(
                vorticity_fft,
                s=(grid_size, grid_size),
            )
            snapshot_times[snapshot_index] = (step + 1) * time_step

    return snapshots, snapshot_times, time.perf_counter() - start_time
