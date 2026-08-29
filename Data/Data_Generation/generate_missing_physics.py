"""Generate the controlled Missing Physics train/test datasets."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import torch

from Data.Data_Generation.navier_stokes import MissingPhysicsBatch, simulate_navier_stokes_2d
from Data.Data_Generation.random_forcing import GaussianRandomField
from project_paths import data_path, ensure_parent
from Scripts.utils import select_device, set_seed


FIELD_NAMES = ("vorticity", "nonlinear", "diffusion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate train/test data for the controlled Missing Physics case."
    )
    parser.add_argument("--trajectories", type=int, default=100)
    parser.add_argument("--train-trajectories", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--output-resolution", type=int, default=64)
    parser.add_argument("--final-time", type=float, default=40.0)
    parser.add_argument("--burn-in-time", type=float, default=20.0)
    parser.add_argument("--time-step", type=float, default=1e-3)
    parser.add_argument("--record-interval-steps", type=int, default=100)
    parser.add_argument("--viscosity", type=float, default=1e-3)
    parser.add_argument("--noise-alpha", type=float, default=0.005)
    parser.add_argument("--noise-kappa", type=int, default=10)
    parser.add_argument("--noise-sigma", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--train-output",
        type=Path,
        default=data_path("Data_MissingPhysics", "train_diffusion_nonlinear.h5"),
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=data_path("Data_MissingPhysics", "test_diffusion_nonlinear.h5"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.trajectories <= 1:
        raise ValueError("trajectories must be greater than one")
    if not 0 < args.train_trajectories < args.trajectories:
        raise ValueError("train-trajectories must be between zero and trajectories")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.resolution <= 0 or args.output_resolution <= 0:
        raise ValueError("spatial resolutions must be positive")
    if args.resolution % args.output_resolution:
        raise ValueError("output-resolution must divide resolution")
    if args.final_time <= 0 or args.time_step <= 0 or args.viscosity <= 0:
        raise ValueError("final-time, time-step, and viscosity must be positive")
    if not 0 <= args.burn_in_time < args.final_time:
        raise ValueError("burn-in-time must be in [0, final-time)")
    if args.record_interval_steps <= 0:
        raise ValueError("record-interval-steps must be positive")
    if args.noise_alpha <= 0 or args.noise_kappa <= 0 or args.noise_sigma <= 0:
        raise ValueError("noise parameters must be positive")
    if args.train_output.resolve() == args.test_output.resolve():
        raise ValueError("train-output and test-output must be different files")


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp")


def _prepare_output(path: Path, overwrite: bool) -> tuple[Path, h5py.File]:
    path = ensure_parent(path)
    temporary = _temporary_path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing dataset: {path}")
    if temporary.exists():
        temporary.unlink()
    return temporary, h5py.File(temporary, "w")


def _create_datasets(
    handle: h5py.File,
    split: str,
    output_resolution: int,
    times: np.ndarray,
    metadata: dict[str, float | int],
) -> None:
    handle.create_dataset("t", data=times)
    for name, value in metadata.items():
        handle.attrs[name] = value
    for field in FIELD_NAMES:
        handle.create_dataset(
            f"{split}_{field}_{output_resolution}",
            shape=(0, output_resolution, output_resolution),
            maxshape=(None, output_resolution, output_resolution),
            chunks=(64, output_resolution, output_resolution),
            dtype=np.float32,
        )


def _append_batch(
    handle: h5py.File,
    split: str,
    batch: MissingPhysicsBatch,
    trajectory_slice: slice,
    output_resolution: int,
) -> None:
    for field in FIELD_NAMES:
        values = getattr(batch, field)[trajectory_slice]
        values = values.permute(0, 3, 1, 2).reshape(-1, output_resolution, output_resolution)
        values = values.detach().cpu().numpy()
        dataset = handle[f"{split}_{field}_{output_resolution}"]
        start = dataset.shape[0]
        dataset.resize(start + values.shape[0], axis=0)
        dataset[start:] = values


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = select_device(args.device)
    output_stride = args.resolution // args.output_resolution

    for output in (args.train_output, args.test_output):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing dataset: {output}")

    train_temporary, train_handle = _prepare_output(args.train_output, args.overwrite)
    test_temporary, test_handle = _prepare_output(args.test_output, args.overwrite)
    handles = (train_handle, test_handle)

    try:
        grid = torch.arange(args.resolution, device=device) / args.resolution
        coordinate_x, coordinate_y = torch.meshgrid(grid, grid, indexing="ij")
        forcing = 0.1 * (
            torch.sin(2.0 * math.pi * (coordinate_x + coordinate_y))
            + torch.cos(2.0 * math.pi * (coordinate_x + coordinate_y))
        )
        initial_condition = GaussianRandomField(
            2,
            args.resolution,
            alpha=2.5,
            tau=7.0,
            device=device,
        )
        stochastic_forcing = {
            "alpha": args.noise_alpha,
            "kappa": args.noise_kappa,
            "sigma": args.noise_sigma,
        }
        metadata = {
            "viscosity": args.viscosity,
            "time_step": args.time_step,
            "burn_in_time": args.burn_in_time,
            "final_time": args.final_time,
            "noise_alpha": args.noise_alpha,
            "noise_kappa": args.noise_kappa,
            "noise_sigma": args.noise_sigma,
            "seed": args.seed,
        }

        initialized = False
        for batch_start in range(0, args.trajectories, args.batch_size):
            batch_stop = min(batch_start + args.batch_size, args.trajectories)
            simulation = simulate_navier_stokes_2d(
                (1.0, 1.0),
                initial_condition.sample(batch_stop - batch_start),
                forcing,
                viscosity=args.viscosity,
                final_time=args.final_time,
                time_step=args.time_step,
                burn_in_time=args.burn_in_time,
                record_interval_steps=args.record_interval_steps,
                stochastic_forcing=stochastic_forcing,
                output_stride=output_stride,
                show_progress=True,
            )
            if not initialized:
                times = simulation.times.cpu().numpy()
                _create_datasets(
                    train_handle,
                    "train",
                    args.output_resolution,
                    times,
                    metadata,
                )
                _create_datasets(
                    test_handle,
                    "test",
                    args.output_resolution,
                    times,
                    metadata,
                )
                initialized = True

            train_count = max(
                0,
                min(batch_stop, args.train_trajectories) - batch_start,
            )
            if train_count:
                _append_batch(
                    train_handle,
                    "train",
                    simulation,
                    slice(0, train_count),
                    args.output_resolution,
                )
            if train_count < batch_stop - batch_start:
                _append_batch(
                    test_handle,
                    "test",
                    simulation,
                    slice(train_count, None),
                    args.output_resolution,
                )
            print(f"Generated trajectories {batch_start + 1}-{batch_stop}")

        for handle in handles:
            handle.flush()
            handle.close()
        os.replace(train_temporary, args.train_output)
        os.replace(test_temporary, args.test_output)
    except BaseException:
        for handle in handles:
            if handle.id.valid:
                handle.close()
        for temporary in (train_temporary, test_temporary):
            if temporary.exists():
                temporary.unlink()
        raise

    print(f"Training data: {args.train_output}")
    print(f"Test data: {args.test_output}")


if __name__ == "__main__":
    main()
