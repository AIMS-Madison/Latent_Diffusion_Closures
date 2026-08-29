"""Conditional score model based on a two-branch Fourier neural operator."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


TensorFunction = Callable[[torch.Tensor], torch.Tensor]


def marginal_prob_std(
    time: torch.Tensor | float,
    sigma: float,
    device_: torch.device | str,
) -> torch.Tensor:
    """Return the perturbation standard deviation for the VE SDE."""
    if sigma <= 1.0:
        raise ValueError("sigma must be greater than one")
    time_tensor = torch.as_tensor(time, dtype=torch.float32, device=device_)
    sigma_tensor = torch.as_tensor(sigma, dtype=time_tensor.dtype, device=time_tensor.device)
    return torch.sqrt(
        (sigma_tensor ** (2.0 * time_tensor) - 1.0)
        / (2.0 * torch.log(sigma_tensor))
    )


def diffusion_coeff(
    time: torch.Tensor | float,
    sigma: float,
    device_: torch.device | str,
) -> torch.Tensor:
    """Return the VE-SDE diffusion coefficient."""
    if sigma <= 1.0:
        raise ValueError("sigma must be greater than one")
    time_tensor = torch.as_tensor(time, dtype=torch.float32, device=device_)
    sigma_tensor = torch.as_tensor(sigma, dtype=time_tensor.dtype, device=time_tensor.device)
    return sigma_tensor**time_tensor


class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim: int, scale: float = 30.0) -> None:
        super().__init__()
        self.W = nn.Parameter(
            torch.randn(embed_dim // 2) * scale,
            requires_grad=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projection = inputs[:, None] * self.W[None, :] * 2.0 * math.pi
        return torch.cat((torch.sin(projection), torch.cos(projection)), dim=-1)


class Dense(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dense(inputs)[..., None, None, None]


class SpectralConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes1, modes2)
        self.weights1 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))

    @staticmethod
    def compl_mul2d(inputs: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", inputs, weights)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.shape[0]
        input_fft = torch.fft.rfft2(inputs)
        output_fft = torch.zeros(
            batch_size,
            self.out_channels,
            inputs.size(-2),
            inputs.size(-1) // 2 + 1,
            dtype=input_fft.dtype,
            device=inputs.device,
        )
        output_fft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            input_fft[:, :, : self.modes1, : self.modes2],
            self.weights1,
        )
        output_fft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            input_fft[:, :, -self.modes1 :, : self.modes2],
            self.weights2,
        )
        return torch.fft.irfft2(output_fft, s=inputs.shape[-2:])


class ConditionalFNOScore(nn.Module):
    """Estimate the conditional score of a noisy target field."""

    def __init__(
        self,
        marginal_prob_std: TensorFunction,
        modes1: int,
        modes2: int,
        width: int,
        padding: int = 0,
        embed_dim: int = 256,
        length: float = 2.0,
    ) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.length = length
        self.padding = padding

        self.fc0 = nn.Linear(3, width)
        self.fc0_w = nn.Linear(3, width)
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        for index in range(4):
            setattr(self, f"conv{index}_x", SpectralConv2d(width, width, modes1, modes1))
            setattr(self, f"conv{index}_w", SpectralConv2d(width, width, modes2, modes2))
            setattr(self, f"w{index}_x", nn.Conv2d(width, width, 1))
            setattr(self, f"w{index}_w", nn.Conv2d(width, width, 1))

        self.dense0 = Dense(embed_dim, width)
        self.transformation_net = nn.Sequential(
            nn.Conv2d(width * 2, width, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 1),
            nn.GELU(),
        )
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.act = lambda value: value * torch.sigmoid(value)
        self.marginal_prob_std = marginal_prob_std

    def _grid(
        self,
        shape: torch.Size,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size, size_x, size_y = shape[:3]
        grid_x = torch.linspace(0.0, self.length, size_x, device=device, dtype=dtype)
        grid_y = torch.linspace(0.0, self.length, size_y, device=device, dtype=dtype)
        grid_x = grid_x.view(1, size_x, 1, 1).expand(batch_size, size_x, size_y, 1)
        grid_y = grid_y.view(1, 1, size_y, 1).expand(batch_size, size_x, size_y, 1)
        return torch.cat((grid_x, grid_y), dim=-1)

    def _lift(self, field: torch.Tensor, layer: nn.Linear) -> torch.Tensor:
        field = field.unsqueeze(-1)
        field = torch.cat(
            (field, self._grid(field.shape, field.device, field.dtype)),
            dim=-1,
        )
        field = layer(field).permute(0, 3, 1, 2)
        return F.pad(field, (0, self.padding, 0, self.padding))

    def _branch(
        self,
        features: torch.Tensor,
        branch: str,
        time_embedding: torch.Tensor | None,
    ) -> torch.Tensor:
        for index in range(4):
            spectral = getattr(self, f"conv{index}_{branch}")(features)
            pointwise = getattr(self, f"w{index}_{branch}")(features)
            features = spectral + pointwise
            if time_embedding is not None:
                features = features + time_embedding
            if index < 3:
                features = F.gelu(features)
        if self.padding:
            features = features[..., : -self.padding, : -self.padding]
        return features

    def forward(
        self,
        time: torch.Tensor,
        noisy_target: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        target_features = self._lift(noisy_target, self.fc0)
        condition_features = self._lift(condition, self.fc0_w)
        embedded_time = self.act(self.embed(time))
        time_features = self.dense0(embedded_time).squeeze(-1)

        target_features = self._branch(target_features, "x", time_features)
        condition_features = self._branch(condition_features, "w", None)
        output = self.transformation_net(
            torch.cat((target_features, condition_features), dim=1)
        )
        output = output.permute(0, 2, 3, 1)
        output = self.fc2(F.gelu(self.fc1(output))).squeeze(-1)
        return output / self.marginal_prob_std(time)[:, None, None]


def denoising_score_matching_loss(
    model: nn.Module,
    target: torch.Tensor,
    condition: torch.Tensor,
    marginal_std: TensorFunction,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute the conditional denoising score-matching objective."""
    random_time = torch.rand(target.shape[0], device=target.device) * (1.0 - eps) + eps
    noise = torch.randn_like(target)
    std = marginal_std(random_time)
    perturbed_target = target + noise * std[:, None, None]
    score = model(random_time, perturbed_target, condition)
    return torch.mean(
        torch.sum((score * std[:, None, None] + noise) ** 2, dim=(1, 2))
    )
