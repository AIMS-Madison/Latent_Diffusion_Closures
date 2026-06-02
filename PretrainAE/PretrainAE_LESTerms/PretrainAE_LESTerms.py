import torch
import torch.nn as nn
import h5py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project_paths import resolve_input_path, resolve_output_path
from training_utils import get_device
from utility import set_seed, fro_err
from AE_Attention import VariationalAutoEncoder, weights_init


# Load and prepare data
device = get_device()

train_name = resolve_input_path(
    "LDM_LES_DATA",
    "LES_NSE/navier_stokes_LES_4096_1e-3.h5",
)
with h5py.File(train_name, 'r') as file:
    train_closure = torch.tensor(file['closure_term'][:10000], device=device)
    test_closure = torch.tensor(file['closure_term'][::100], device=device)

train_loader = torch.utils.data.DataLoader(train_closure, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_closure, batch_size=10, shuffle=False)

# Usage example
model = VariationalAutoEncoder().to(device)
pretrained_model = resolve_input_path(
    "LDM_PRETRAINED_CLOSURE_AE",
    "JointAE/Joint_AE_Closure_6416.pth",
)
model.load_state_dict(torch.load(pretrained_model, map_location=device))
# model.apply(weights_init)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=100)
criterion = nn.MSELoss()

# Training loop with early stopping
num_epochs = 100
patience = 100
best_val_loss = float('inf')
counter = 0

recon_loss_history = torch.zeros(num_epochs)
var_loss_history = torch.zeros(num_epochs)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_fro = 0
    for batch in train_loader:
        inputs = batch.to(device)
        latent = model.encode(inputs)
        decoded = model.decode(latent)

        flattened_latent_x = latent.view(latent.shape[0], -1)
        latent_mean = flattened_latent_x.mean(dim=0)
        latent_var = flattened_latent_x.var(dim=0, unbiased=True)
        kl_divergence = 0.5 * (latent_var + latent_mean ** 2 - 1 - torch.log(
            latent_var + 1e-8))
        var_loss = kl_divergence.mean() * 1e-3

        recon_loss = criterion(decoded, inputs)  # Reconstruction loss (e.g., MSE)
        loss = recon_loss + var_loss # Total loss (reconstruction + regularization)

        recon_loss_history[epoch] = recon_loss

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_fro += fro_err(inputs, decoded)

    train_loss /= len(train_loader)
    train_fro /= len(train_loader)

    # Test
    model.eval()
    test_loss = 0
    test_fro = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, inputs).item()
            test_fro += fro_err(inputs, outputs)

    test_loss /= len(test_loader)
    test_fro /= len(test_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}] |',
          f'Last LR: {scheduler.get_last_lr()[0]:.6f} |',
          f'Train Loss: {train_loss:.6f} |',
          f'Train Fro Error: {train_fro:.6f} |',
          f'Recon Loss: {recon_loss:.6f} |',
            f'Var Loss: {var_loss:.6f} |',
          f'Test Loss: {test_loss:.6f}',
          f'Test Fro Error: {test_fro:.6f}')

    # Learning rate scheduler step
    scheduler.step(test_loss)

    # Early stopping based on validation loss
    if test_loss < best_val_loss:
        best_val_loss = test_loss
        counter = 0
        model_path = resolve_output_path("PretrainAE/PretrainAE_LESTerms/PretrainAE_6416_LESClosure_v2.pth")
        torch.save(model.state_dict(), model_path)
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping")
            break
# Load best model
model.load_state_dict(torch.load(model_path, map_location=device))




# Evaluation on test set
model.eval()
test_loss = 0
test_re = 0

with torch.no_grad():
    test_batch = test_closure[-2:-1].to(device)
    test_output  = model(test_batch)
    encoded = model.encode(test_batch)
    test_re = fro_err(test_batch, test_output)
    test_mse = criterion(test_output, test_batch).item()

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
sns.heatmap(test_closure[-1].cpu().numpy(), ax=axs[0], cbar=True, cmap="rocket")
axs[0].set_title("Original")
axs[0].axis("off")
sns.heatmap(test_output[-1].cpu().numpy(), ax=axs[1], cbar=True, cmap="rocket")
axs[1].set_title("Reconstructed")
axs[1].axis("off")
sns.heatmap(encoded[-1].cpu().numpy(), ax=axs[2], cbar=True, cmap="rocket")
axs[2].set_title("Encoded")
axs[2].axis("off")
plt.show()
