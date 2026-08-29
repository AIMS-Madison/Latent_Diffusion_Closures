from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm.auto import trange

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Scripts.missing_physics.common import (
    JOINT_LCDM_CHECKPOINT,
    JOINT_NONLINEAR_AE,
    JOINT_VORTICITY_AE,
    LATENT_SCORE_CONFIG,
    TRAIN_DATA,
    build_score_model,
    make_pair_loader,
    make_sde_functions,
    save_weights,
)
from Scripts.models.autoencoder import FieldAutoencoder
from Scripts.models.diffusion import denoising_score_matching_loss
from Scripts.utils import select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train joint L-CDM for Missing Physics.")
    parser.add_argument("--train-data", type=Path, default=TRAIN_DATA)
    parser.add_argument("--diffusion-output", type=Path, default=JOINT_LCDM_CHECKPOINT)
    parser.add_argument("--nonlinear-ae-output", type=Path, default=JOINT_NONLINEAR_AE)
    parser.add_argument("--vorticity-ae-output", type=Path, default=JOINT_VORTICITY_AE)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scheduler-step", type=int, default=100)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--nonlinear-reconstruction-weight", type=float, default=10.0)
    parser.add_argument("--vorticity-reconstruction-weight", type=float, default=0.1)
    parser.add_argument("--score-weight", type=float, default=0.1)
    parser.add_argument("--kl-weight", type=float, default=0.01)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def latent_kl_loss(latent: torch.Tensor) -> torch.Tensor:
    flattened = latent.flatten(start_dim=1)
    mean = flattened.mean(dim=0)
    variance = flattened.var(dim=0, unbiased=False)
    return 0.5 * torch.mean(
        variance + mean.square() - 1.0 - torch.log(variance + 1e-8)
    )


def main() -> None:
    args = parse_args()
    if min(args.epochs, args.batch_size, args.scheduler_step) < 1:
        raise ValueError("epochs, batch-size, and scheduler-step must be positive")
    if args.learning_rate <= 0 or not 0 < args.scheduler_gamma <= 1:
        raise ValueError("learning-rate must be positive and scheduler-gamma in (0, 1]")
    loss_weights = (
        args.nonlinear_reconstruction_weight,
        args.vorticity_reconstruction_weight,
        args.score_weight,
        args.kl_weight,
    )
    if any(weight < 0 for weight in loss_weights) or not any(loss_weights):
        raise ValueError("loss weights must be nonnegative and not all zero")
    set_seed(args.seed)
    device = select_device(args.device)
    loader = make_pair_loader(
        args.train_data,
        "train",
        args.batch_size,
        limit=args.max_samples,
    )
    marginal_std, _ = make_sde_functions(device)
    nonlinear_autoencoder = FieldAutoencoder().to(device)
    vorticity_autoencoder = FieldAutoencoder().to(device)
    score_model = build_score_model(LATENT_SCORE_CONFIG, marginal_std, device)
    optimizer = torch.optim.Adam(
        list(score_model.parameters())
        + list(nonlinear_autoencoder.parameters())
        + list(vorticity_autoencoder.parameters()),
        lr=args.learning_rate,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.scheduler_step,
        gamma=args.scheduler_gamma,
    )

    progress = trange(1, args.epochs + 1, desc="Joint L-CDM training")
    for epoch in progress:
        score_model.train()
        nonlinear_autoencoder.train()
        vorticity_autoencoder.train()
        totals = {"loss": 0.0, "score": 0.0, "nonlinear": 0.0, "vorticity": 0.0, "kl": 0.0}
        sample_count = 0

        for nonlinear, vorticity in loader:
            nonlinear = nonlinear.to(device, non_blocking=True)
            vorticity = vorticity.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            latent_nonlinear = nonlinear_autoencoder.encode(nonlinear)
            latent_vorticity = vorticity_autoencoder.encode(vorticity)
            reconstructed_nonlinear = nonlinear_autoencoder.decode(latent_nonlinear)
            reconstructed_vorticity = vorticity_autoencoder.decode(latent_vorticity)

            nonlinear_reconstruction = F.mse_loss(reconstructed_nonlinear, nonlinear)
            vorticity_reconstruction = F.mse_loss(reconstructed_vorticity, vorticity)
            score_loss = denoising_score_matching_loss(
                score_model,
                latent_nonlinear,
                latent_vorticity,
                marginal_std,
            )
            kl_loss = latent_kl_loss(latent_nonlinear)
            loss = (
                args.nonlinear_reconstruction_weight * nonlinear_reconstruction
                + args.vorticity_reconstruction_weight * vorticity_reconstruction
                + args.score_weight * score_loss
                + args.kl_weight * kl_loss
            )
            loss.backward()
            optimizer.step()

            batch_size = nonlinear.shape[0]
            totals["loss"] += loss.item() * batch_size
            totals["score"] += score_loss.item() * batch_size
            totals["nonlinear"] += nonlinear_reconstruction.item() * batch_size
            totals["vorticity"] += vorticity_reconstruction.item() * batch_size
            totals["kl"] += kl_loss.item() * batch_size
            sample_count += batch_size

        scheduler.step()
        averages = {name: value / sample_count for name, value in totals.items()}
        progress.set_description(
            f"Epoch {epoch}/{args.epochs} | total {averages['loss']:.4e} | "
            f"score {averages['score']:.4e} | H recon {averages['nonlinear']:.4e} | "
            f"w recon {averages['vorticity']:.4e} | KL {averages['kl']:.4e}"
        )

    diffusion_path = save_weights(score_model, args.diffusion_output)
    nonlinear_path = save_weights(nonlinear_autoencoder, args.nonlinear_ae_output)
    vorticity_path = save_weights(vorticity_autoencoder, args.vorticity_ae_output)
    print(f"Saved diffusion model to {diffusion_path}")
    print(f"Saved nonlinear autoencoder to {nonlinear_path}")
    print(f"Saved vorticity autoencoder to {vorticity_path}")


if __name__ == "__main__":
    main()
