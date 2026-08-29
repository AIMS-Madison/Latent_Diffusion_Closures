"""Train one reconstruction-only autoencoder for the Missing Physics case."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Scripts.missing_physics.common import (
    NONLINEAR_AE,
    TEST_DATA,
    TRAIN_DATA,
    VORTICITY_AE,
    load_field,
    save_weights,
)
from Scripts.models.autoencoder import FieldAutoencoder, initialize_weights
from Scripts.utils import relative_frobenius_error, select_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a reconstruction-only field autoencoder.",
    )
    parser.add_argument("--field", choices=("nonlinear", "vorticity"), required=True)
    parser.add_argument("--train-data", type=Path, default=TRAIN_DATA)
    parser.add_argument("--test-data", type=Path, default=TEST_DATA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--scheduler-patience", type=int, default=50)
    parser.add_argument("--early-stopping-patience", type=int, default=100)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def evaluate(
    model: FieldAutoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_mse = 0.0
    total_relative_error = 0.0
    sample_count = 0
    with torch.no_grad():
        for inputs in loader:
            inputs = inputs.to(device, non_blocking=True)
            reconstruction = model(inputs)
            batch_size = inputs.shape[0]
            total_mse += criterion(reconstruction, inputs).item() * batch_size
            total_relative_error += (
                relative_frobenius_error(inputs, reconstruction).item() * batch_size
            )
            sample_count += batch_size
    return total_mse / sample_count, total_relative_error / sample_count


def main() -> None:
    args = parse_args()
    if min(
        args.epochs,
        args.batch_size,
        args.scheduler_patience,
        args.early_stopping_patience,
    ) < 1:
        raise ValueError("epochs, batch-size, and patience values must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")

    set_seed(args.seed)
    device = select_device(args.device)
    output = args.output or (
        NONLINEAR_AE if args.field == "nonlinear" else VORTICITY_AE
    )
    train_data = load_field(
        args.train_data,
        "train",
        args.field,
        limit=args.max_train_samples,
    )
    test_data = load_field(
        args.test_data,
        "test",
        args.field,
        limit=args.max_test_samples,
    )
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )

    model = FieldAutoencoder().to(device)
    model.apply(initialize_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.scheduler_patience,
    )
    criterion = nn.MSELoss()
    best_validation_loss = float("inf")
    stale_epochs = 0

    print(f"Training {args.field} autoencoder on {device}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for inputs in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(inputs)
            loss = criterion(reconstruction, inputs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.shape[0]
            sample_count += inputs.shape[0]

        train_loss = total_loss / sample_count
        validation_loss, validation_relative_error = evaluate(
            model,
            test_loader,
            criterion,
            device,
        )
        scheduler.step(validation_loss)
        print(
            f"Epoch {epoch:04d}/{args.epochs} | "
            f"train MSE {train_loss:.5e} | "
            f"validation MSE {validation_loss:.5e} | "
            f"validation RE {validation_relative_error:.5e}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            stale_epochs = 0
            save_weights(model, output)
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"Best checkpoint: {Path(output).resolve()}")


if __name__ == "__main__":
    main()
