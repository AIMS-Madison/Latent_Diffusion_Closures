# Latent Diffusion Closures

Official implementation of the paper ["Stochastic and Non-local Closure Modeling for Nonlinear Dynamical Systems via Latent Score-based Generative Models"](https://doi.org/10.1016/j.jcp.2026.115082).

Xinghao Dong, Huchen Yang, and Jin-Long Wu

---

## Overview

This repository contains the Missing Physics experiments for learning the unresolved nonlinear term in a two-dimensional stochastic vorticity equation. The code supports three conditional diffusion strategies:

- **P-CDM:** score-based closure generation in the 64-by-64 physical space.
- **Conventional L-CDM:** separate autoencoder pretraining followed by score-model training in a 16-by-16 latent space.
- **Joint L-CDM:** joint optimization of the two autoencoders and latent score model.

The runnable notebook demonstrates the vorticity-field comparison from Figure 7, the ensemble-mean relative-error diagnostic from Table 5, and the energy-spectrum comparison from Figure 10. Training and evaluation entry points report numeric results only; visualization is kept in the notebook.

This release focuses on the controlled Missing Physics problem and includes its data generator, pretrained checkpoints, and running demo. The same training and evaluation workflow can be used for the under-resolved LES study in Appendix E, or adapted to other compatible datasets, by supplying the corresponding data and updating the data-loading configuration. Model and training hyperparameters may need to be retuned for a new dataset. The LES dataset and LES-specific pretrained checkpoints are not distributed in this repository.

## Repository Structure

```text
Data/
  Data_Generation/             Missing Physics data simulator
Scripts/
  missing_physics/             Training, encoding, and evaluation entry points
  models/                      Autoencoder and conditional score-based model
  solvers/                     Vorticity rollout solver
  sampling.py                  Reverse-SDE sampler
  training.py                  Shared score-model training loop
  utils.py                     Runtime and metric utilities
Trained_Models/
  AE/Missing_Physics/          Autoencoder checkpoints
  DM/Missing_Physics/          Diffusion-model checkpoints
project_paths.py               Repository-relative path configuration
running_demo.ipynb             P-CDM and Joint L-CDM simulation demo
```

## Setup

Clone the repository, retrieve the Git LFS checkpoints, and install the Python dependencies:

```bash
git clone https://github.com/AIMS-Madison/Latent_Diffusion_Closures.git
cd Latent_Diffusion_Closures
git lfs pull
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

CUDA is recommended for training and for the full notebook simulation. Every command also accepts `--device cpu` or another PyTorch device string.

## Data

The full datasets used in the paper are available from the authors upon reasonable request. The controlled Missing Physics datasets can also be generated with the included scripts:

```bash
python -m Data.Data_Generation.generate_missing_physics --device cuda
```

## Training

Run all commands from the repository root.

### P-CDM

```bash
python -m Scripts.missing_physics.train_pcdm --device cuda
```

### Conventional L-CDM

```bash
python -m Scripts.missing_physics.train_autoencoder --field nonlinear --device cuda
python -m Scripts.missing_physics.train_autoencoder --field vorticity --device cuda
python -m Scripts.missing_physics.encode_latent_data --device cuda
python -m Scripts.missing_physics.train_conventional_lcdm --device cuda
```

### Joint L-CDM

```bash
python -m Scripts.missing_physics.train_joint_lcdm --device cuda
```

## Evaluation

Evaluate a supplied or newly trained checkpoint without creating plots or tables:

```bash
python -m Scripts.missing_physics.evaluate --method pcdm --device cuda
python -m Scripts.missing_physics.evaluate --method joint-lcdm --device cuda
python -m Scripts.missing_physics.evaluate --method conventional-lcdm --device cuda
```

## Running Demo

Open [running_demo.ipynb](running_demo.ipynb) from the repository root and execute the cells in order. The notebook loads the supplied Missing Physics checkpoints, runs P-CDM and Joint L-CDM rollouts, and reports Figure 7-style vorticity fields, Table 5-style ensemble-mean relative errors, and Figure 10-style energy spectra. The notebook fixes the random seed at 42 to improve run-to-run repeatability. Because the sampling seeds used for the paper were not retained, results should be compared at the level of error magnitude and qualitative behavior rather than exact numerical values.

## Citation

```bibtex
@article{dong2026stochastic,
  title   = {Stochastic and Non-local Closure Modeling for Nonlinear Dynamical Systems via Latent Score-based Generative Models},
  author  = {Dong, Xinghao and Yang, Huchen and Wu, Jin-Long},
  journal = {Journal of Computational Physics},
  volume  = {563},
  pages   = {115082},
  year    = {2026},
  doi     = {10.1016/j.jcp.2026.115082}
}
```
