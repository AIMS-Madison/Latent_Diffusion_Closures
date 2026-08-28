"""Reusable diffusion sampling routines."""

from __future__ import annotations

from collections.abc import Callable

import torch


def diffusion_sampler(
    condition: torch.Tensor,
    score_model: torch.nn.Module,
    spatial_dim: int,
    marginal_prob_std: Callable[[torch.Tensor], torch.Tensor],
    diffusion_coeff: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int,
    num_steps: int,
    time_noises: torch.Tensor,
    device: torch.device | str,
    *,
    initial_x: torch.Tensor | None = None,
    stochastic: bool = True,
) -> torch.Tensor:
    """Sample a field with Euler-Maruyama integration of the reverse SDE."""
    device = torch.device(device)
    t = torch.ones(batch_size, device=device) * time_noises[0]
    if initial_x is None:
        x = torch.randn(batch_size, spatial_dim, spatial_dim, device=device)
        x = x * marginal_prob_std(t)[:, None, None]
    else:
        x = initial_x.to(device)

    mean_x = x
    with torch.no_grad():
        for i in range(num_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_noises[i]
            step_size = time_noises[i] - time_noises[i + 1]
            g = diffusion_coeff(batch_time_step)
            grad = score_model(batch_time_step, x, condition)

            mean_x = x + (g ** 2)[:, None, None] * grad * step_size
            if stochastic:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
            x = mean_x + torch.sqrt(step_size) * g[:, None, None] * noise

    return mean_x


def diffusion_sampler_with_score_error(
    target: torch.Tensor,
    condition: torch.Tensor,
    score_model: torch.nn.Module,
    spatial_dim: int,
    marginal_prob_std: Callable[[torch.Tensor], torch.Tensor],
    diffusion_coeff: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int,
    num_steps: int,
    time_noises: torch.Tensor,
    device: torch.device | str,
    error_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a field and record score error against the analytic score."""
    device = torch.device(device)
    t = torch.ones(batch_size, device=device) * time_noises[0]
    x = torch.randn(batch_size, spatial_dim, spatial_dim, device=device)
    x = x * marginal_prob_std(t)[:, None, None]
    rel_err = torch.zeros(num_steps, device=device)
    mean_x = x

    with torch.no_grad():
        for i in range(num_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_noises[i]
            real_score = -(x - target) / marginal_prob_std(batch_time_step)[:, None, None] ** 2
            step_size = time_noises[i] - time_noises[i + 1]
            g = diffusion_coeff(batch_time_step)
            grad = score_model(batch_time_step, x, condition)

            mean_x = x + (g ** 2)[:, None, None] * grad * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None] * torch.randn_like(x)
            rel_err[i] = error_fn(real_score, grad)

    return mean_x, rel_err
