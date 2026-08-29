from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Scripts.missing_physics.common import (
    CONVENTIONAL_LCDM_CHECKPOINT,
    JOINT_LCDM_CHECKPOINT,
    JOINT_NONLINEAR_AE,
    JOINT_VORTICITY_AE,
    LATENT_SCORE_CONFIG,
    NONLINEAR_AE,
    PCDM_CHECKPOINT,
    PHYSICAL_SCORE_CONFIG,
    TEST_DATA,
    VORTICITY_AE,
    build_score_model,
    load_pair,
    load_weights,
    make_sde_functions,
)
from Scripts.models.autoencoder import FieldAutoencoder
from Scripts.sampling import euler_maruyama_sampler
from Scripts.utils import (
    get_sigmas_karras,
    mean_squared_error,
    relative_frobenius_error,
    select_device,
    set_seed,
    synchronize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Missing Physics closure model.")
    parser.add_argument(
        "--method",
        choices=("pcdm", "conventional-lcdm", "joint-lcdm"),
        required=True,
    )
    parser.add_argument("--test-data", type=Path, default=TEST_DATA)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--sampling-steps", type=int, default=10)
    parser.add_argument("--time-min", type=float, default=1e-3)
    parser.add_argument("--time-max", type=float, default=0.4)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--nonlinear-autoencoder", type=Path)
    parser.add_argument("--vorticity-autoencoder", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def latent_paths(method: str) -> tuple[Path, Path, Path]:
    if method == "conventional-lcdm":
        return CONVENTIONAL_LCDM_CHECKPOINT, NONLINEAR_AE, VORTICITY_AE
    return JOINT_LCDM_CHECKPOINT, JOINT_NONLINEAR_AE, JOINT_VORTICITY_AE


def main() -> None:
    args = parse_args()
    if args.samples < 1 or args.sampling_steps < 1:
        raise ValueError("samples and sampling-steps must be positive")
    if not 0 < args.time_min < args.time_max <= 1:
        raise ValueError("diffusion times must satisfy 0 < time-min < time-max <= 1")
    set_seed(args.seed)
    device = select_device(args.device)
    nonlinear, vorticity = load_pair(
        args.test_data,
        "test",
        limit=args.samples,
    )
    nonlinear = nonlinear.to(device)
    vorticity = vorticity.to(device)
    marginal_std, coefficient = make_sde_functions(device)
    schedule = get_sigmas_karras(
        args.sampling_steps,
        args.time_min,
        args.time_max,
        device=device,
    )

    if args.method == "pcdm":
        checkpoint = args.checkpoint or PCDM_CHECKPOINT
        score_model = build_score_model(PHYSICAL_SCORE_CONFIG, marginal_std, device)
        load_weights(score_model, checkpoint, device)
        score_model.eval()
        condition = vorticity
        decoder = None
    else:
        default_checkpoint, default_nonlinear_ae, default_vorticity_ae = latent_paths(args.method)
        checkpoint = args.checkpoint or default_checkpoint
        nonlinear_autoencoder = FieldAutoencoder().to(device)
        vorticity_autoencoder = FieldAutoencoder().to(device)
        load_weights(
            nonlinear_autoencoder,
            args.nonlinear_autoencoder or default_nonlinear_ae,
            device,
        )
        load_weights(
            vorticity_autoencoder,
            args.vorticity_autoencoder or default_vorticity_ae,
            device,
        )
        nonlinear_autoencoder.eval()
        vorticity_autoencoder.eval()
        score_model = build_score_model(LATENT_SCORE_CONFIG, marginal_std, device)
        load_weights(score_model, checkpoint, device)
        score_model.eval()
        with torch.no_grad():
            condition = vorticity_autoencoder.encode(vorticity)
        decoder = nonlinear_autoencoder

    synchronize(device)
    start = time.perf_counter()
    generated = euler_maruyama_sampler(
        condition,
        score_model,
        marginal_std,
        coefficient,
        schedule,
    )
    if decoder is not None:
        with torch.no_grad():
            generated = decoder.decode(generated)
    synchronize(device)
    elapsed = time.perf_counter() - start

    print(f"Method: {args.method}")
    print(f"Samples: {generated.shape[0]}")
    print(f"Sampling time: {elapsed:.4f} s")
    print(f"MSE: {mean_squared_error(nonlinear, generated).item():.6e}")
    print(
        "Relative Frobenius error: "
        f"{relative_frobenius_error(nonlinear, generated).item():.6e}"
    )


if __name__ == "__main__":
    main()
