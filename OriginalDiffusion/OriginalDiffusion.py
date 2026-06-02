import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import torch

import numpy as np
from torch.optim import Adam
from functools import partial
from tqdm import trange
from project_paths import resolve_input_path, resolve_output_path
from sampling_utils import diffusion_sampler
from training_utils import create_ticks_labels, get_device
from utility import set_seed, fro_err, mse_err, get_sigmas_karras
from DiffusionModel import (marginal_prob_std, diffusion_coeff, FNO2d_Orig, loss_fn)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
plt.rc("text", usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"


device = get_device()

# Load the data
train_name = resolve_input_path(
    "LDM_LES_DATA",
    "LES_NSE/navier_stokes_LES_4096_1e-3.h5",
)
test_name = resolve_input_path(
    "LDM_LES_TEST_DATA",
    "LES_NSE/navier_stokes_LES_4096_1e-3.h5",
)

with h5py.File(train_name, 'r') as file:
    train_closure = torch.tensor(file['closure_term'][:10000], device=device)
    train_vorticity = torch.tensor(file['filtered_vorticity'][:10000], device=device)

with h5py.File(test_name, 'r') as file:
    test_closure = torch.tensor(file['closure_term'][::100], device=device)
    test_vorticity = torch.tensor(file['filtered_vorticity'][::100], device=device)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_closure, train_vorticity),
    batch_size=100, shuffle=True
)

################################
######## Model Training ########
################################
sigma = 30
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

modes = 6
width = 40
epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

model = FNO2d_Orig(marginal_prob_std_fn, modes, modes, width, padding = 0, embed_dim = 512, length = 1).to(device)
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

tqdm_epoch = trange(epochs)

loss_history = []
rel_err_history = []

set_seed(42)
for epoch in tqdm_epoch:
    model.train()
    avg_loss = 0.
    num_items = 0
    for x, w in train_loader:
        x, w = x.to(device), w.to(device)
        optimizer.zero_grad()
        loss, _, _ = loss_fn(model, x, w, None, marginal_prob_std_fn, sparse=False)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
        # relative_loss = torch.mean(torch.norm(score - real_score, 2, dim=(1, 2))
        #                            / torch.norm(real_score, 2, dim=(1, 2)))
        # rel_err.append(relative_loss.item())
    scheduler.step()
    avg_loss_epoch = avg_loss / num_items
    # relative_loss_epoch = np.mean(rel_err)
    loss_history.append(avg_loss_epoch)
    # rel_err_history.append(relative_loss_epoch)
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))

savepath = resolve_output_path("OriginalDiffusion/LES_Closure_PCDM_1e-3.pth")
torch.save(model.state_dict(), savepath)


model.load_state_dict(torch.load(savepath, map_location=device))

sde_time_min = 1e-3
sde_time_max = 0.1
sample_steps = 100
sample_batch_size = 100

time_noises = get_sigmas_karras(sample_steps, sde_time_min, sde_time_max, device=device)

sample_spatial_dim = 64

sampler = partial(
    diffusion_sampler,
    spatial_dim=sample_spatial_dim,
    marginal_prob_std=marginal_prob_std_fn,
    diffusion_coeff=diffusion_coeff_fn,
    batch_size=sample_batch_size,
    num_steps=sample_steps,
    time_noises=time_noises,
    device=device,
)


with torch.no_grad():
    test_sample = sampler(test_vorticity[:sample_batch_size], model)

fro_err_ = fro_err(test_closure[:sample_batch_size], test_sample)
mse_err_ = mse_err(test_closure[:sample_batch_size], test_sample)






index = 10
import time
start = time.time()
with torch.no_grad():
    test_sample = sampler(test_vorticity[index:index+1].repeat(sample_batch_size, 1, 1), model)
end = time.time()
print('Time elapsed: {}'.format(end - start))

rel_err_col = torch.zeros(sample_batch_size, device=device)
for i in range(sample_batch_size):
    rel_err_col[i] = fro_err(test_closure[index:index+1], test_sample[i:i+1])

plt.plot(rel_err_col.cpu().numpy())
plt.show()

test_sample_mean = test_sample.mean(dim=0, keepdim=True)
fro_sample = fro_err(test_closure[index:index+1], test_sample_mean[0:1])
mse_sample = mse_err(test_closure[index:index+1], test_sample[index:index+1])


### Plot and save
set_seed(13)

data1 = test_closure[index:index + 1, :, :].repeat(sample_batch_size, 1, 1).cpu()
data2 = test_sample.cpu()
data3 = np.abs(data1 - data2)

# Initialize the plot with 4 rows and 4 columns
fig, axs = plt.subplots(3, 4, figsize=(20, 15), constrained_layout=True)
fs = 28
plt.rcParams.update({'font.size': fs})

ticks_1, tick_labels_1 = create_ticks_labels(data1.shape[1])
ticks_2, tick_labels_2 = create_ticks_labels(data2.shape[1])
ticks_3, tick_labels_3 = create_ticks_labels(data3.shape[1])

# Randomly sample indices equal to the number of columns (4) for clarity
indices = [torch.randint(0, 100, (1,)).item() for _ in range(4)]

# Define color scale parameters
max_val = 0.2
min_val = -0.3
err_max = 0.01
err_min = 0
cbar_ticks = np.linspace(min_val, max_val, 6)
cbar_ticks_err = np.linspace(err_min, err_max, 6)
cbar_ticks_contour = np.linspace(err_min, err_max, 6)

# Plot heatmaps and contour plots
for i, idx in enumerate(indices):
    j = i % 4  # Column index

    # --- Row 1: Truth Heatmap ---
    truth = data1[idx, ...].cpu().numpy()
    sns.heatmap(
        truth,
        ax=axs[0, j],
        cmap='rocket',
        cbar=(j == 3),  # Show colorbar only on the last column
        vmax=max_val,
        vmin=min_val,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks}
    )
    axs[0, j].set_title(r"\text{Truth }" + str(j + 1))
    axs[0, j].set_xticks(ticks_1)
    axs[0, j].set_yticks(ticks_1)
    axs[0, j].set_xticklabels(tick_labels_1, rotation=0)
    axs[0, j].set_yticklabels(tick_labels_1, rotation=0)
    axs[0, j].invert_yaxis()

    # --- Row 2: Generated Heatmap ---
    generated = data2[idx, ...].cpu().numpy()
    sns.heatmap(
        generated,
        ax=axs[1, j],
        cmap='rocket',
        cbar=(j == 3),
        vmax=max_val,
        vmin=min_val,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks}
    )

    axs[1, j].set_title(r"\text{Generated }" + str(j + 1))
    axs[1, j].set_xticks(ticks_2)
    axs[1, j].set_yticks(ticks_2)
    axs[1, j].set_xticklabels(tick_labels_2, rotation=0)
    axs[1, j].set_yticklabels(tick_labels_2, rotation=0)
    axs[1, j].invert_yaxis()

    # --- Row 3: Error Heatmap ---
    error = data3[idx, ...].cpu().numpy()
    ax_contour = axs[2, j]
    # Define the grid coordinates
    S = error.shape[0]
    x = np.arange(S)
    y = np.arange(S)
    X, Y = np.meshgrid(x, y)

    # Create filled contour plot using matplotlib
    contour = ax_contour.contourf(
        X, Y, error,
        levels=cbar_ticks_contour,  # Six levels to match cbar_ticks_err
        cmap='rocket',
        vmin=err_min,
        vmax=err_max
    )

    # Add colorbar only on the last column
    if j == 3:
        cbar_contour = fig.colorbar(
            contour,
            ax=ax_contour,
            format='%.3f'
        )

    ax_contour.set_title(r"\text{Error Contour }" + str(j + 1))
    ax_contour.set_xticks(ticks_3)
    ax_contour.set_yticks(ticks_3)
    ax_contour.set_xticklabels(tick_labels_3, rotation=0)
    ax_contour.set_yticklabels(tick_labels_3, rotation=0)

# Adjust tick parameters for all axes
for ax in axs.flat:
    ax.tick_params(axis='both', which='major', labelsize=fs)


# Adjust layout and save the plot
plt.subplots_adjust(right=0.85, hspace=0.3, wspace=0.5)
# plt.show()
plt.savefig(
    resolve_output_path("figures/LESPhysicalCDM.png"),
    dpi=300,
    bbox_inches='tight'
)
