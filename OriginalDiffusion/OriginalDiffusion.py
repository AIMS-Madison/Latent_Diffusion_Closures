import sys
import h5py
import torch
import numpy as np
from torch.optim import Adam
from functools import partial
from tqdm import trange
from utility import set_seed, fro_err, mse_err, get_sigmas_karras
from DiffusionModel import (marginal_prob_std, diffusion_coeff, FNO2d_Orig, loss_fn)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
plt.rc("text", usetex=True)
mpl.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

sys.path.append('C:\\UWMadisonResearch\\Joint_LDM')
# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device('cuda')
else:
    print("CUDA is not available.")
    device = torch.device('cpu')

# Load the data

train_name = 'C:\\UWMadisonResearch\\Joint_LDM\\Data\\train_diffusion_nonlinear_sto_v2.h5'
with h5py.File(train_name, 'r') as file:
    train_vorticity = torch.tensor(file['train_vorticity_64'][:10000], device=device)
    train_nonlinear = torch.tensor(file['train_nonlinear_64'][:10000], device=device)

test_name = 'C:\\UWMadisonResearch\\Joint_LDM\\Data\\test_diffusion_nonlinear_sto_v2.h5'
with h5py.File(test_name, 'r') as file:
    test_vorticity = torch.tensor(file['test_vorticity_64'][:], device=device)
    test_nonlinear = torch.tensor(file['test_nonlinear_64'][:], device=device)


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_nonlinear,
                                                                          train_vorticity),
                                                                            batch_size=100, shuffle=True)


with torch.no_grad():
    current_max_dist = 0
    lam = 1e-6
    for i, (x, w) in enumerate(train_loader):
        x = x.to(device)
        x_ = x.view(x.shape[0], -1)
        max_dist = torch.cdist(x_, x_).max().item()

        if current_max_dist < max_dist:
            current_max_dist = max_dist
        print(current_max_dist)
    print('Final, max eucledian distance: {}'.format(current_max_dist))

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
        x, w = x.cuda(), w.cuda()
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
torch.save(model.state_dict(), '../Convection_NoSparse_NoAE_4096_sto_v2.pth')



model.load_state_dict(torch.load('OriginalDiffusion/Convection_NoSparse_NoAE_4096_sto.pth'))

sde_time_min = 1e-3
sde_time_max = 0.1
sample_steps = 10
sample_batch_size = 100

time_noises = get_sigmas_karras(sample_steps, sde_time_min, sde_time_max, device=device)
time_noises = torch.linspace(sde_time_max, sde_time_min, sample_steps+1, device=device)

def sampler(vorticity_condition,
           score_model,
            spatial_dim,
            marginal_prob_std,
           diffusion_coeff,
           batch_size,
           num_steps,
           time_noises,
           device):
    t = torch.ones(batch_size, device=device) * time_noises[0]
    init_x = torch.randn(batch_size, spatial_dim, spatial_dim, device=device) * marginal_prob_std(t)[:, None, None]
    x = init_x
    with (torch.no_grad()):
        for i in range(num_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_noises[i]
            step_size = time_noises[i] - time_noises[i + 1]
            g = diffusion_coeff(batch_time_step)
            grad = score_model(batch_time_step, x, vorticity_condition)
            mean_x = x + (g ** 2)[:, None, None] * grad * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None] * torch.randn_like(x)
    return mean_x

sample_spatial_dim = 64

sampler = partial(sampler,
                  spatial_dim=sample_spatial_dim,
                marginal_prob_std = marginal_prob_std_fn,
                diffusion_coeff = diffusion_coeff_fn,
                batch_size = sample_batch_size,
                num_steps = sample_steps,
                time_noises = time_noises,
                device = device)

import time
start = time.time()
with torch.no_grad():
    test_sample = sampler(test_vorticity[:sample_batch_size], model)
end = time.time()
print('Time elapsed: {}'.format(end - start))


fro_sample = fro_err(test_nonlinear[:sample_batch_size], test_sample)
mse_sample = mse_err(test_nonlinear[:sample_batch_size], test_sample)





### Plot and save
set_seed(13)

data1 = test_nonlinear[:sample_batch_size, :, :].cpu()
data2 = test_sample.cpu()
data3 = np.abs(data1 - data2)

# Initialize the plot with 4 rows and 4 columns
fig, axs = plt.subplots(3, 4, figsize=(20, 15), constrained_layout=True)
fs = 28
plt.rcParams.update({'font.size': fs})

# Define tick positions and labels
def create_ticks_labels(size, step=20):
    ticks = np.arange(0, size, step * size / 64)
    tick_labels = [str(int(tick)) for tick in ticks]
    return ticks, tick_labels

ticks_1, tick_labels_1 = create_ticks_labels(data1.shape[1])
ticks_2, tick_labels_2 = create_ticks_labels(data2.shape[1])
ticks_3, tick_labels_3 = create_ticks_labels(data3.shape[1])

# Randomly sample indices equal to the number of columns (4) for clarity
indices = [torch.randint(0, data1.shape[0], (1,)).item() for _ in range(4)]

# Define color scale parameters
max_val = 0.7
min_val = -0.8
err_max = 0.1
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
            format='%.2f'
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
plt.savefig(
    'C:\\UWMadisonResearch\\Joint_LDM\\Plots\\PhysicalLDM.png',
    dpi=300,
    bbox_inches='tight'
)