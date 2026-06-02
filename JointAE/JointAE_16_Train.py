#!/usr/bin/env python
"""Train the joint autoencoder and latent diffusion model."""

from functools import partial
from pathlib import Path
import sys
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib as mpl
import numpy as np
import torch
from torch.optim import Adam
from tqdm import trange

from AE_Attention import VariationalAutoEncoder
from DiffusionModel import diffusion_coeff, FNO2d_Orig, loss_fn, marginal_prob_std
from project_paths import resolve_input_path, resolve_output_path
from training_utils import get_device
from utility import fro_err, set_seed


mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

np.set_printoptions(suppress=False, formatter={"float": "{:.2e}".format})
torch.set_printoptions(sci_mode=True)
warnings.filterwarnings("ignore")

device = get_device()


def load_data():
    train_name = resolve_input_path(
        "LDM_LES_DATA",
        "LES_NSE/navier_stokes_LES_4096_1e-3.h5",
    )

    print(f"Loading training data from {train_name}")
    with h5py.File(train_name, "r") as file:
        train_closure = torch.tensor(file["closure_term"][:40000], device=device)
        train_vorticity = torch.tensor(file["filtered_vorticity"][:40000], device=device)

    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_closure, train_vorticity),
        batch_size=100,
        shuffle=True,
    )


sigma = 30
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

modes = 4
width = 20
padding = 0
epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

AEG_model = VariationalAutoEncoder().to(device)
AEW_model = VariationalAutoEncoder().to(device)
diffusion_model = FNO2d_Orig(
    marginal_prob_std_fn,
    modes,
    modes,
    width,
    padding,
    embed_dim=256,
    length=1,
).to(device)

# To initialize from pretrained weights, set an environment variable and load it here.
# AEG_path = resolve_input_path("LDM_PRETRAINED_CLOSURE_AE", "PretrainAE/AE_6416_vorticity_reg_sto_v5.pth")
# AEG_model.load_state_dict(torch.load(AEG_path, map_location=device))

optimizer = Adam(
    list(diffusion_model.parameters()) + list(AEW_model.parameters()) + list(AEG_model.parameters()),
    lr=learning_rate,
)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=scheduler_step,
    gamma=scheduler_gamma,
)


def local_closeness_loss(x, z, k=5):
    batch_size = x.shape[0]
    x_flat = x.view(batch_size, -1)
    z_flat = z.view(batch_size, -1)
    d_x = torch.cdist(x_flat, x_flat, p=2)
    d_z = torch.cdist(z_flat, z_flat, p=2)
    loss = 0.0
    for i in range(batch_size):
        dx = d_x[i].clone()
        dx[i] = float("inf")
        idxs = torch.topk(dx, k, largest=False).indices
        loss += ((d_z[i, idxs] - d_x[i, idxs]) ** 2).mean()
    return loss / batch_size


def train():
    train_loader = load_data()
    loss_history = []
    tqdm_epoch = trange(epochs, desc="Training")
    for epoch in tqdm_epoch:
        diffusion_model.train()
        AEW_model.train()
        AEG_model.train()
        total_loss = 0.0
        num_items = 0

        for x, w in train_loader:
            x, w = x.to(device), w.to(device)
            optimizer.zero_grad()

            latent_x = AEG_model.encode(x)
            recon_x = AEG_model.decode(latent_x)
            fro_x = fro_err(x, recon_x)

            flattened_latent_x = latent_x.view(latent_x.shape[0], -1)
            latent_mean = flattened_latent_x.mean(dim=0)
            latent_var = flattened_latent_x.var(dim=0, unbiased=True)
            kl_divergence = 0.5 * (latent_var + latent_mean ** 2 - 1 - torch.log(latent_var + 1e-8))
            var_loss = kl_divergence.mean() * 0.1

            latent_w = AEW_model.encode(w)
            recon_w = AEW_model.decode(latent_w)
            fro_w = fro_err(w, recon_w)

            recon_loss_x = torch.nn.MSELoss()(recon_x, x) * 100
            recon_loss_w = torch.nn.MSELoss()(recon_w, w)

            score_loss, _, _ = loss_fn(diffusion_model, latent_x, latent_w, None, marginal_prob_std_fn, sparse=False)

            loss = score_loss + recon_loss_x + recon_loss_w + var_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        scheduler.step()
        avg_loss = total_loss / num_items
        loss_history.append(avg_loss)
        tqdm_epoch.set_description(
            f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.5f}, "
            f"Fro Nonlinear: {fro_x:.5f}, Fro Vorticity: {fro_w:.5f}"
        )

        print(f"recon_loss_x: {recon_loss_x.item():.5f}, recon_loss_w: {recon_loss_w.item():.5f}")
        print(f"score_loss: {score_loss.item():.5f}, fro_x: {fro_x.item():.5f}, fro_w: {fro_w.item():.5f}")

    torch.save(
        diffusion_model.state_dict(),
        resolve_output_path("JointAE/Joint_diffusion_LES_6416.pth"),
    )
    torch.save(
        AEG_model.state_dict(),
        resolve_output_path("JointAE/Joint_AE_Closure_6416.pth"),
    )
    torch.save(
        AEW_model.state_dict(),
        resolve_output_path("JointAE/Joint_AE_Vorticity_6416.pth"),
    )

    print("Training complete. Models saved.")


if __name__ == "__main__":
    set_seed(42)
    train()
