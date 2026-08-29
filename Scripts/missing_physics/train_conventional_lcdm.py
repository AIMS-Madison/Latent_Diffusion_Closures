from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Scripts.missing_physics.common import (
    CONVENTIONAL_LCDM_CHECKPOINT,
    ENCODED_TRAIN_DATA,
    LATENT_SCORE_CONFIG,
    build_score_model,
    make_pair_loader,
    make_sde_functions,
    save_weights,
)
from Scripts.models.diffusion import denoising_score_matching_loss
from Scripts.training import fit_score_model
from Scripts.utils import select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train conventional two-phase L-CDM for Missing Physics.",
    )
    parser.add_argument("--train-data", type=Path, default=ENCODED_TRAIN_DATA)
    parser.add_argument("--output", type=Path, default=CONVENTIONAL_LCDM_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scheduler-step", type=int, default=100)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.batch_size, args.scheduler_step) < 1:
        raise ValueError("epochs, batch-size, and scheduler-step must be positive")
    if args.learning_rate <= 0 or not 0 < args.scheduler_gamma <= 1:
        raise ValueError("learning-rate must be positive and scheduler-gamma in (0, 1]")
    set_seed(args.seed)
    device = select_device(args.device)
    loader = make_pair_loader(
        args.train_data,
        "train",
        args.batch_size,
        encoded=True,
        limit=args.max_samples,
    )
    marginal_std, _ = make_sde_functions(device)
    model = build_score_model(LATENT_SCORE_CONFIG, marginal_std, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.scheduler_step,
        gamma=args.scheduler_gamma,
    )

    def batch_loss(model_, nonlinear, vorticity):
        return denoising_score_matching_loss(
            model_,
            nonlinear,
            vorticity,
            marginal_std,
        )

    fit_score_model(
        model,
        loader,
        batch_loss,
        optimizer,
        scheduler,
        args.epochs,
        device,
    )
    output = save_weights(model, args.output)
    print(f"Saved conventional L-CDM checkpoint to {output}")


if __name__ == "__main__":
    main()
