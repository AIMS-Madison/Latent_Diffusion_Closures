"""Convolutional autoencoder with residual and self-attention blocks."""

from __future__ import annotations

import torch
from torch import nn


def initialize_weights(module: nn.Module) -> None:
    """Initialize convolutional layers with Kaiming-normal weights."""
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(
            module.weight,
            mode="fan_out",
            nonlinearity="leaky_relu",
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class SelfAttention(nn.Module):
    """Self-attention over all spatial locations."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.qkv_project = nn.Linear(channels, 3 * channels, bias=False)
        self.mha = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.out_project = nn.Linear(channels, channels, bias=False)

    def _attention(self, inputs: torch.Tensor) -> torch.Tensor:
        query, key, value = self.qkv_project(inputs).chunk(3, dim=-1)
        attended, _ = self.mha(query, key, value, need_weights=False)
        return self.out_project(attended)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError("attention inputs must have shape (batch, channels, height, width)")

        batch, channels, height, width = inputs.shape
        sequence = inputs.flatten(2).transpose(1, 2).contiguous()
        output = self._attention(sequence)
        return output.transpose(1, 2).reshape(batch, channels, height, width).contiguous()


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.GroupNorm(32, channels),
            SelfAttention(channels=channels, num_heads=num_heads),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.attention(inputs)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layer = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.res = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layer(inputs) + self.res(inputs)


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 1, latent_channels: int = 1) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            ResidualBlock(64, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock(128, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256, 256),
            AttentionBlock(256, 8),
            ResidualBlock(256, 256),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.Conv2d(256, latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.encoder(inputs)


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 1, latent_channels: int = 1) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 256),
            AttentionBlock(256),
            ResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            ResidualBlock(256, 128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            ResidualBlock(128, 64),
            nn.GroupNorm(32, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(inputs)


class FieldAutoencoder(nn.Module):
    """Map a single-channel 64-by-64 field to 16-by-16 and back."""

    def __init__(self, in_channels: int = 1, latent_channels: int = 1) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels, latent_channels)
        self.decoder = Decoder(in_channels, latent_channels)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (batch, height, width)")
        return self.encoder(inputs.unsqueeze(1)).squeeze(1)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3:
            raise ValueError("latent must have shape (batch, height, width)")
        return self.decoder(latent.unsqueeze(1)).squeeze(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))
