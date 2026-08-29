"""Shared data and checkpoint contracts for the Missing Physics experiments."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import h5py
import torch
from torch.utils.data import DataLoader, TensorDataset

from project_paths import data_path, ensure_parent, model_path
from Scripts.models.diffusion import (
    ConditionalFNOScore,
    diffusion_coeff,
    marginal_prob_std,
)


Split = Literal["train", "test"]
Field = Literal["nonlinear", "vorticity"]

TRAIN_DATA = data_path("Data_MissingPhysics", "train_diffusion_nonlinear.h5")
TEST_DATA = data_path("Data_MissingPhysics", "test_diffusion_nonlinear.h5")
ENCODED_TRAIN_DATA = data_path(
    "Data_MissingPhysics",
    "train_diffusion_nonlinear_encoded_recon_only.h5",
)
ENCODED_TEST_DATA = data_path(
    "Data_MissingPhysics",
    "test_diffusion_nonlinear_encoded_recon_only.h5",
)

NONLINEAR_AE = model_path(
    "AE",
    "Missing_Physics",
    "AE_Nonlinear_ReconOnly.pth",
)
VORTICITY_AE = model_path(
    "AE",
    "Missing_Physics",
    "AE_Vorticity_ReconOnly.pth",
)
JOINT_NONLINEAR_AE = model_path(
    "AE",
    "Missing_Physics",
    "Joint_AE_Nonlinear_DM.pth",
)
JOINT_VORTICITY_AE = model_path(
    "AE",
    "Missing_Physics",
    "Joint_AE_Vorticity_DM.pth",
)
PCDM_CHECKPOINT = model_path("DM", "Missing_Physics", "P-CDM.pth")
CONVENTIONAL_LCDM_CHECKPOINT = model_path(
    "DM",
    "Missing_Physics",
    "L-CDM_ReconOnly.pth",
)
JOINT_LCDM_CHECKPOINT = model_path("DM", "Missing_Physics", "Joint_DM.pth")

VE_SIGMA = 30.0


@dataclass(frozen=True)
class ScoreModelConfig:
    modes: int
    width: int
    embed_dim: int
    padding: int = 0
    domain_length: float = 1.0


PHYSICAL_SCORE_CONFIG = ScoreModelConfig(
    modes=6,
    width=40,
    embed_dim=512,
)
LATENT_SCORE_CONFIG = ScoreModelConfig(
    modes=4,
    width=20,
    embed_dim=256,
)


def field_key(split: Split, field: Field, *, encoded: bool = False) -> str:
    suffix = "_encoded" if encoded else "_64"
    return f"{split}_{field}{suffix}"


def _validate_field(dataset: h5py.Dataset, key: str, *, encoded: bool) -> None:
    if dataset.ndim != 3:
        raise ValueError(f"{key} must have shape (samples, height, width)")
    expected_size = 16 if encoded else 64
    if dataset.shape[1:] != (expected_size, expected_size):
        raise ValueError(
            f"{key} must use a {expected_size}-by-{expected_size} spatial grid"
        )
    if dataset.shape[0] == 0:
        raise ValueError(f"{key} cannot be empty")


def load_field(
    path: Path,
    split: Split,
    field: Field,
    *,
    encoded: bool = False,
    limit: int | None = None,
) -> torch.Tensor:
    """Load one HDF5 field into CPU memory as float32."""
    path = Path(path)
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive when provided")
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset: {path}")
    key = field_key(split, field, encoded=encoded)
    with h5py.File(path, "r") as handle:
        if key not in handle:
            raise KeyError(f"Dataset {path} does not contain {key!r}")
        dataset = handle[key]
        _validate_field(dataset, key, encoded=encoded)
        stop = dataset.shape[0] if limit is None else min(limit, dataset.shape[0])
        values = dataset[:stop]
    return torch.from_numpy(values).float()


def load_pair(
    path: Path,
    split: Split,
    *,
    encoded: bool = False,
    limit: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load nonlinear targets and their paired vorticity conditions."""
    nonlinear = load_field(
        path,
        split,
        "nonlinear",
        encoded=encoded,
        limit=limit,
    )
    vorticity = load_field(
        path,
        split,
        "vorticity",
        encoded=encoded,
        limit=limit,
    )
    if nonlinear.shape != vorticity.shape:
        raise ValueError(
            "Nonlinear targets and vorticity conditions must have identical shapes"
        )
    return nonlinear, vorticity


def make_pair_loader(
    path: Path,
    split: Split,
    batch_size: int,
    *,
    encoded: bool = False,
    limit: int | None = None,
    shuffle: bool = True,
) -> DataLoader:
    nonlinear, vorticity = load_pair(
        path,
        split,
        encoded=encoded,
        limit=limit,
    )
    return DataLoader(
        TensorDataset(nonlinear, vorticity),
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )


def make_sde_functions(
    device: torch.device,
    sigma: float = VE_SIGMA,
):
    marginal = partial(marginal_prob_std, sigma=sigma, device_=device)
    coefficient = partial(diffusion_coeff, sigma=sigma, device_=device)
    return marginal, coefficient


def build_score_model(
    config: ScoreModelConfig,
    marginal_std,
    device: torch.device,
) -> ConditionalFNOScore:
    return ConditionalFNOScore(
        marginal_std,
        modes1=config.modes,
        modes2=config.modes,
        width=config.width,
        padding=config.padding,
        embed_dim=config.embed_dim,
        length=config.domain_length,
    ).to(device)


def load_weights(
    module: torch.nn.Module,
    path: Path,
    device: torch.device,
) -> None:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    module.load_state_dict(state)


def save_weights(module: torch.nn.Module, path: Path) -> Path:
    path = ensure_parent(path)
    torch.save(module.state_dict(), path)
    return path
