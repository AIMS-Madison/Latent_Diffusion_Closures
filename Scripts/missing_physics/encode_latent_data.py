from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import h5py
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from project_paths import ensure_parent
from Scripts.missing_physics.common import (
    ENCODED_TEST_DATA,
    ENCODED_TRAIN_DATA,
    NONLINEAR_AE,
    TEST_DATA,
    TRAIN_DATA,
    VORTICITY_AE,
    field_key,
    load_weights,
)
from Scripts.models.autoencoder import FieldAutoencoder
from Scripts.utils import select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encode physical fields for conventional L-CDM training.",
    )
    parser.add_argument("--train-data", type=Path, default=TRAIN_DATA)
    parser.add_argument("--test-data", type=Path, default=TEST_DATA)
    parser.add_argument("--train-output", type=Path, default=ENCODED_TRAIN_DATA)
    parser.add_argument("--test-output", type=Path, default=ENCODED_TEST_DATA)
    parser.add_argument("--nonlinear-autoencoder", type=Path, default=NONLINEAR_AE)
    parser.add_argument("--vorticity-autoencoder", type=Path, default=VORTICITY_AE)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def encode_file(
    source_path: Path,
    output_path: Path,
    split: str,
    nonlinear_autoencoder: FieldAutoencoder,
    vorticity_autoencoder: FieldAutoencoder,
    batch_size: int,
    device: torch.device,
    overwrite: bool,
) -> None:
    nonlinear_key = field_key(split, "nonlinear")
    vorticity_key = field_key(split, "vorticity")
    encoded_nonlinear_key = field_key(split, "nonlinear", encoded=True)
    encoded_vorticity_key = field_key(split, "vorticity", encoded=True)
    output_path = ensure_parent(output_path)
    source_path = Path(source_path).expanduser().resolve()
    if source_path == output_path:
        raise ValueError("source and output paths must be different")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite encoded data: {output_path}")
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    try:
        with h5py.File(source_path, "r") as source:
            for key in (nonlinear_key, vorticity_key):
                if key not in source:
                    raise KeyError(f"Dataset {source_path} does not contain {key!r}")
            if source[nonlinear_key].shape != source[vorticity_key].shape:
                raise ValueError("Nonlinear and vorticity arrays must have matching shapes")

            sample_count = source[nonlinear_key].shape[0]
            if sample_count == 0:
                raise ValueError(f"Dataset {source_path} is empty")
            with h5py.File(temporary_path, "w") as output:
                output.attrs["source"] = source_path.name
                nonlinear_output = output.create_dataset(
                    encoded_nonlinear_key,
                    shape=(sample_count, 16, 16),
                    dtype="float32",
                    chunks=(min(batch_size, sample_count), 16, 16),
                )
                vorticity_output = output.create_dataset(
                    encoded_vorticity_key,
                    shape=(sample_count, 16, 16),
                    dtype="float32",
                    chunks=(min(batch_size, sample_count), 16, 16),
                )

                for start in range(0, sample_count, batch_size):
                    stop = min(start + batch_size, sample_count)
                    nonlinear = torch.from_numpy(source[nonlinear_key][start:stop])
                    vorticity = torch.from_numpy(source[vorticity_key][start:stop])
                    nonlinear = nonlinear.float().to(device)
                    vorticity = vorticity.float().to(device)
                    with torch.no_grad():
                        encoded_nonlinear = nonlinear_autoencoder.encode(nonlinear)
                        encoded_vorticity = vorticity_autoencoder.encode(vorticity)
                    nonlinear_output[start:stop] = encoded_nonlinear.cpu().numpy()
                    vorticity_output[start:stop] = encoded_vorticity.cpu().numpy()
                    print(f"{split}: encoded samples {start}:{stop}")
        os.replace(temporary_path, output_path)
    except BaseException:
        if temporary_path.exists():
            temporary_path.unlink()
        raise


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    for output in (args.train_output, args.test_output):
        if output.exists() and not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite encoded data: {output}")
    device = select_device(args.device)
    nonlinear_autoencoder = FieldAutoencoder().to(device)
    vorticity_autoencoder = FieldAutoencoder().to(device)
    load_weights(nonlinear_autoencoder, args.nonlinear_autoencoder, device)
    load_weights(vorticity_autoencoder, args.vorticity_autoencoder, device)
    nonlinear_autoencoder.eval()
    vorticity_autoencoder.eval()

    encode_file(
        args.train_data,
        args.train_output,
        "train",
        nonlinear_autoencoder,
        vorticity_autoencoder,
        args.batch_size,
        device,
        args.overwrite,
    )
    encode_file(
        args.test_data,
        args.test_output,
        "test",
        nonlinear_autoencoder,
        vorticity_autoencoder,
        args.batch_size,
        device,
        args.overwrite,
    )


if __name__ == "__main__":
    main()
