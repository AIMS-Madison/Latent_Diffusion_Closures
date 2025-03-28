import sys
import os
import h5py
import torch
import numpy as np
from matplotlib.pyplot import savefig
from torch.optim import Adam
from functools import partial
from tqdm import trange
from utility import set_seed, fro_err, mse_err, get_sigmas_karras
from DiffusionModel import (marginal_prob_std, diffusion_coeff, FNO2d_Orig, loss_fn)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
plt.rc("text", usetex=True)
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

onedrive_path = '/mnt/c/Users/dongx/OneDriveUWM'

# Load the data
train_name = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "Data",
                          "train_diffusion_nonlinear_sto_v.h5")
test_name = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "Data",
                         "test_diffusion_nonlinear_sto_v4.h5")

with h5py.File(train_name, 'r') as file:
    train_nonlinear = torch.tensor(file['train_nonlinear_64'][:], device=device)
    train_vorticity = torch.tensor(file['train_vorticity_64'][:], device=device)

with h5py.File(test_name, 'r') as file:
    test_nonlinear = torch.tensor(file['test_nonlinear_64'][:], device=device)
    test_vorticity = torch.tensor(file['test_vorticity_64'][:], device=device)
    test_forcing = torch.tensor(file['test_forcing_64'][:], device=device)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_nonlinear, train_vorticity),
    batch_size=100, shuffle=True
)

beta = 5e-5
scaled_forcing = test_forcing * beta
nonlinear_nonoise = test_nonlinear - scaled_forcing

test_nonlinear_onesnap = nonlinear_nonoise[1:2, :, :]
test_nonlinear_onesnap_noise = torch.zeros(1000, 64, 64, device=device)
for i in range(1000):
    test_nonlinear_onesnap_noise[i:i + 1, :, :] = test_nonlinear_onesnap + scaled_forcing[i:i + 1, :, :]

def total_variance_torch(data):
    """
    Compute the total variance (trace of the covariance matrix) of data using PyTorch.

    Parameters:
    data (torch.Tensor): Data tensor of shape (N, 4096)

    Returns:
    torch.Tensor: The total variance of the data.
    """
    N = data.shape[0]
    # Compute the mean over samples (dim=0)
    mean = torch.mean(data, dim=0, keepdim=True)
    # Center the data
    data_centered = data - mean
    # Compute covariance matrix using the unbiased estimator
    cov_matrix = (data_centered.T @ data_centered) / (N - 1)
    # Total variance is the trace of the covariance matrix
    total_var = torch.trace(cov_matrix)
    return total_var

# Flatten to (N, 4096)
H_torch_flat = test_nonlinear_onesnap_noise.view(1000, -1)


total_var_H_torch = total_variance_torch(H_torch_flat)


print("Total Variance of H (PyTorch):", total_var_H_torch.item())


var_field = torch.var(test_nonlinear_onesnap_noise, dim=0)
var_field_ranked_index = torch.sort(var_field.view(4096, -1), dim=0, descending=True)[0]

plt.plot(var_field_ranked_index.cpu().numpy())
plt.title('Variance of H')
plt.xlabel('Rank')
plt.ylabel('Variance')
plt.show()

rel_err = fro_err(test_nonlinear_onesnap, test_nonlinear_onesnap_noise[1:2])






from skimage.metrics import structural_similarity as ssim


def structure_preservation(H, noise):
    """Compute how much structural information is preserved"""
    deterministic = H - noise
    ssim_values = []

    for i in range(H.shape[0]):
        ssim_val = ssim(H[i].cpu().numpy(), deterministic[i].cpu().numpy(), data_range=H[i].cpu().numpy().max() - H[i].cpu().numpy().min())
        ssim_values.append(ssim_val)

    return np.mean(ssim_values)

structure_preservation(test_nonlinear, scaled_forcing)

noise_fro_err = fro_err(nonlinear_nonoise, test_nonlinear)
noise_mse_err = mse_err(nonlinear_nonoise, test_nonlinear)

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

savepath = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "OriginalDiffusion", "Convection_NoSparse_NoAE_4096_sto_v5.pth")
torch.save(model.state_dict(), savepath)


model_name = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "OriginalDiffusion",
                         "Convection_NoSparse_NoAE_4096_sto_v2.pth")
model.load_state_dict(torch.load(model_name))

sde_time_min = 1e-3
sde_time_max = 0.4
sample_steps = 10
sample_batch_size = 1000

time_noises = get_sigmas_karras(sample_steps, sde_time_min, sde_time_max, device=device)

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
    return x

sample_spatial_dim = 64

sampler = partial(sampler,
                  spatial_dim=sample_spatial_dim,
                marginal_prob_std = marginal_prob_std_fn,
                diffusion_coeff = diffusion_coeff_fn,
                batch_size = sample_batch_size,
                num_steps = sample_steps,
                time_noises = time_noises,
                device = device)


with torch.no_grad():
    test_sample = sampler(test_vorticity[:sample_batch_size], model)

fro_err_ = fro_err(test_nonlinear[:sample_batch_size], test_sample)
mse_err_ = mse_err(test_nonlinear[:sample_batch_size], test_sample)






index = 10
import time
start = time.time()
with torch.no_grad():
    test_sample = sampler(test_vorticity[index:index+1].repeat(sample_batch_size, 1, 1), model)
end = time.time()
print('Time elapsed: {}'.format(end - start))

rel_err_col = torch.zeros(sample_batch_size, device=device)
for i in range(sample_batch_size):
    rel_err_col[i] = fro_err(test_nonlinear[index:index+1], test_sample[i:i+1])

plt.plot(rel_err_col.cpu().numpy())
plt.show()

test_sample_mean = test_sample.mean(dim=0, keepdim=True)
fro_sample = fro_err(test_nonlinear[index:index+1], test_sample_mean[0:1])
fro_sample_nonoise = fro_err(test_nonlinear[index:index+1]-1e-4 * test_forcing[index:index+1], test_sample_mean[0:1])
mse_sample = mse_err(test_nonlinear[index:index+1], test_sample[index:index+1])

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift
import matplotlib.colors as colors


def calculate_fluctuation_spectrum(samples):
    """
    Calculate the energy spectrum of fluctuations for a set of 2D samples.

    Parameters:
    ----------
    samples : ndarray
        Samples array of shape (n_samples, nx, ny)

    Returns:
    -------
    wavenumbers : ndarray
        1D array of wavenumbers
    spectrum : ndarray
        1D array of fluctuation energy values corresponding to wavenumbers
    """
    n_samples, nx, ny = samples.shape

    # Calculate mean across all samples
    mean_field = np.mean(samples, axis=0)

    # Compute 2D FFT of fluctuations for each sample and average the power spectra
    k_spectrum = np.zeros((nx, ny))

    for i in range(n_samples):
        # Compute fluctuation (difference from mean)
        fluctuation = samples[i] - mean_field

        # 2D FFT of the fluctuation
        fft_fluctuation = fftshift(fft2(fluctuation))

        # Power spectrum (squared magnitude of FFT)
        power = np.abs(fft_fluctuation) ** 2

        # Accumulate
        k_spectrum += power

    # Average over samples
    k_spectrum /= n_samples

    # Create wavenumber grid
    kx = np.fft.fftfreq(nx, d=1.0)
    ky = np.fft.fftfreq(ny, d=1.0)

    # Create 2D grids for kx, ky
    kx = fftshift(kx)
    ky = fftshift(ky)
    kx_grid, ky_grid = np.meshgrid(kx, ky)

    # Calculate magnitude of wavenumber vector at each point
    k_grid = np.sqrt(kx_grid ** 2 + ky_grid ** 2)

    # Convert to 1D spectrum by averaging over rings of constant k
    # Create bins of wavenumbers
    dk = 1.0 / max(nx, ny)  # Wavenumber resolution
    k_max = np.max(k_grid)
    k_bins = np.arange(0, k_max + dk, dk)

    # Initialize spectrum
    spectrum = np.zeros(len(k_bins) - 1)

    # Bin the energy
    for i in range(len(k_bins) - 1):
        k_lower = k_bins[i]
        k_upper = k_bins[i + 1]

        # Find all points in this wavenumber range
        mask = (k_grid >= k_lower) & (k_grid < k_upper)

        if np.any(mask):
            spectrum[i] = np.mean(k_spectrum[mask])

    # Wavenumbers for plotting (center of bins)
    wavenumbers = (k_bins[:-1] + k_bins[1:]) / 2

    return wavenumbers, spectrum

# Example usage (replace with your actual data):
analyze_model_fluctuations(test_sample.cpu().numpy(), None, None)










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
plt.show()
plt.savefig(
    'C:\\UWMadisonResearch\\Joint_LDM\\Plots\\PhysicalLDM.png',
    dpi=300,
    bbox_inches='tight'
)