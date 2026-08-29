"""Reverse-SDE sampling for physical and latent conditional diffusion models."""

from __future__ import annotations

from collections.abc import Callable

import torch


TensorFunction = Callable[[torch.Tensor], torch.Tensor]


def euler_maruyama_sampler(
    condition: torch.Tensor,
    score_model: torch.nn.Module,
    marginal_prob_std: TensorFunction,
    diffusion_coeff: TensorFunction,
    time_steps: torch.Tensor,
    *,
    initial_sample: torch.Tensor | None = None,
    stochastic: bool = True,
) -> torch.Tensor:
    """Draw one conditional sample per item in ``condition``."""
    if condition.ndim != 3:
        raise ValueError("condition must have shape (batch, height, width)")
    if time_steps.ndim != 1 or time_steps.numel() < 2:
        raise ValueError("time_steps must contain at least two values")
    if torch.any(time_steps[:-1] <= time_steps[1:]):
        raise ValueError("time_steps must be strictly decreasing")

    device = condition.device
    time_steps = time_steps.to(device=device, dtype=condition.dtype)
    batch_size = condition.shape[0]
    initial_time = torch.full(
        (batch_size,),
        time_steps[0],
        device=device,
        dtype=condition.dtype,
    )
    if initial_sample is None:
        sample = torch.randn_like(condition)
        sample = sample * marginal_prob_std(initial_time)[:, None, None]
    else:
        if initial_sample.shape != condition.shape:
            raise ValueError("initial_sample and condition must have the same shape")
        sample = initial_sample.to(device=device, dtype=condition.dtype)

    mean_sample = sample
    with torch.no_grad():
        for current_time, next_time in zip(time_steps[:-1], time_steps[1:]):
            batch_time = torch.full(
                (batch_size,),
                current_time,
                device=device,
                dtype=condition.dtype,
            )
            step_size = current_time - next_time
            coefficient = diffusion_coeff(batch_time)
            score = score_model(batch_time, sample, condition)
            mean_sample = (
                sample
                + coefficient[:, None, None] ** 2 * score * step_size
            )
            noise = torch.randn_like(sample) if stochastic else torch.zeros_like(sample)
            sample = (
                mean_sample
                + torch.sqrt(step_size) * coefficient[:, None, None] * noise
            )
    return mean_sample
