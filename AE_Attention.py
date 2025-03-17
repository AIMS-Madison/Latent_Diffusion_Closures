import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from einops import rearrange, repeat

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class SelfAttention(nn.Module):
    def __init__(
            self,
            channels: int,
            num_heads: int = 4
    ):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.qkv_project = nn.Linear(channels, 3 * channels, bias=False)
        self.mha = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )
        self.out_project = nn.Linear(channels, channels, bias=False)

    def _get_attention_score(self, x: torch.Tensor):
        q, k, v = self.qkv_project(x).chunk(3, dim=-1)
        attention_value, _ = self.mha(q, k, v)
        output = self.out_project(attention_value)
        return output

    def forward(
        self,
        x: torch.Tensor,
        reshape_out: bool = True
    ):
        if len(x.shape) == 4:  # image 4 dim: [B, C, H, W]
            B, C, H, W = x.shape
            x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
            output = self._get_attention_score(x)
            if reshape_out:
                return rearrange(output, 'b (h w) c -> b c h w', h=H, w=W).contiguous()
            else:
                return output
        return self._get_attention_score(x)

class VAEAttentionBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 4
    ):
        super().__init__()
        self.attention = nn.Sequential(
            nn.GroupNorm(32, channels),
            SelfAttention(channels=channels, num_heads=num_heads)
        )

    def forward(self, x: torch.Tensor):
        return x + self.attention(x)

class VAEResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.res = nn.Identity() if (
            in_channels == out_channels
        ) else nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor):
        return self.layer(x) + self.res(x)

class VAEEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 8
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            VAEResidualBlock(64, 128),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            VAEResidualBlock(128, 256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            VAEResidualBlock(256, 256),
            VAEAttentionBlock(256, 8),
            VAEResidualBlock(256, 256),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            nn.Conv2d(256, latent_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.encoder(x)

class VAEDecoder(nn.Module):
    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 8
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, kernel_size=3, padding=1),
            VAEResidualBlock(256, 256),
            VAEAttentionBlock(256),
            VAEResidualBlock(256, 256),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            VAEResidualBlock(256, 128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            VAEResidualBlock(128, 64),
            nn.GroupNorm(32, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.decoder(x)

class VariationalAutoEncoder(pl.LightningModule):
    def __init__(self,
                 in_channels: int = 1,
                 latent_channels: int = 1,
                 num_embeds: int = 256,
                 ):
        super().__init__()
        self.encoder = VAEEncoder(in_channels, latent_channels)
        self.decoder = VAEDecoder(in_channels, latent_channels)

    def encode(self, x: torch.Tensor):
        x_latent = self.encoder(x.unsqueeze(1)).squeeze(1)
        return x_latent

    def decode(self, x_latent: torch.Tensor):
        return self.decoder(x_latent.unsqueeze(1)).squeeze(1)

    def forward(self, x: torch.Tensor):
        x_latent = self.encode(x)
        return self.decode(x_latent)
