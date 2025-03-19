### Standard Libraries
import os
import warnings
import time

### Scientific Computing & Deep Learning Libraries
import numpy as np
import torch
import h5py
from torch.optim import Adam
from functools import partial
from tqdm import tqdm, trange

### Visualization Libraries
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

### Custom Modules
from DiffusionModel import marginal_prob_std, diffusion_coeff, FNO2d_Orig, loss_fn
from utility import get_sigmas_karras, fro_err, mse_err, set_seed
from AE_Attention import VariationalAutoEncoder

### Configure Matplotlib for LaTeX Rendering (if available)
plt.rc("text", usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

### Configure NumPy & PyTorch
np.set_printoptions(suppress=False, formatter={'float': '{:.2e}'.format})
torch.set_printoptions(sci_mode=True)
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Environment Configuration
# ------------------------------------------------------------------

### Get OneDrive Path from Environment Variables
onedrive_path = os.getenv("ONEDRIVE_PATH")
if not onedrive_path:
    raise ValueError("OneDrive path not found. Please check the environment variable!")

### Check CUDA Availability
def get_device():
    if torch.cuda.is_available():
        print("✅ CUDA is available. Using GPU.")
        return torch.device('cuda')
    else:
        print("❌ CUDA is not available. Using CPU.")
        return torch.device('cpu')

device = get_device()

# ------------------------------------------------------------------
# Load Data
# ------------------------------------------------------------------

### Load test dataset from HDF5 file
test_name = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "Data", "test_diffusion_nonlinear_sto_v4.h5")
with h5py.File(test_name, 'r') as file:
    test_nonlinear = torch.tensor(file['test_nonlinear_64'][:], device=device)
    test_vorticity = torch.tensor(file['test_vorticity_64'][:], device=device)
    test_forcing = torch.tensor(file['test_forcing_64'][:], device=device)

# ------------------------------------------------------------------
# Model Configuration
# ------------------------------------------------------------------

sigma = 30
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

### Model parameters
modes = 4
width = 20
padding = 0

### Initialize Models
AEG_model = VariationalAutoEncoder().to(device)
AEW_model = VariationalAutoEncoder().to(device)
diffusion_model = FNO2d_Orig(marginal_prob_std_fn, modes, modes, width, padding, embed_dim=256, length=1).to(device)

### Load Pretrained Weights
diffusion_model_save = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "JointAE", "Joint_diffusion_6416_sto_v2.pth")
AEG_model_save = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "JointAE", "Joint_AE_Nonlinear_6416_sto_v2.pth")
AEW_model_save = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "JointAE", "Joint_AE_Vorticity_6416_sto_v2.pth")

diffusion_model_save = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "PretrainAE", "PretrainAE_Diffusion_reg_sto_v2.pth")
AEG_model_save = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "PretrainAE", "AE_6416_nonlinear_reg_sto_v2.pth")
AEW_model_save = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "PretrainAE", "AE_6416_vorticity_reg_sto_v2.pth")


AEG_model.load_state_dict(torch.load(AEG_model_save))
AEW_model.load_state_dict(torch.load(AEW_model_save))
diffusion_model.load_state_dict(torch.load(diffusion_model_save))
### Set Models to Evaluation Mode
AEG_model.eval()
AEW_model.eval()
diffusion_model.eval()

# ------------------------------------------------------------------
# Sampler Function
# ------------------------------------------------------------------

### Define time noises for the sampling process
sde_time_min = 1e-3
sde_time_max = 0.1
steps = 10
time_noises = get_sigmas_karras(steps, sde_time_min, sde_time_max, device=device)


### Sampler function using Stochastic Differential Equations (SDE)
def sampler(vorticity_condition, score_model, spatial_dim, marginal_prob_std, diffusion_coeff, batch_size, num_steps, time_noises, device):
    t = torch.ones(batch_size, device=device) * time_noises[0]
    x = torch.randn(batch_size, spatial_dim, spatial_dim, device=device) * marginal_prob_std(t)[:, None, None]

    with torch.no_grad():
        for i in range(num_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_noises[i]
            step_size = time_noises[i] - time_noises[i + 1]
            g = diffusion_coeff(batch_time_step)
            grad = score_model(batch_time_step, x, vorticity_condition)

            mean_x = x + (g ** 2)[:, None, None] * grad * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None] * torch.randn_like(x)

    return x


# ------------------------------------------------------------------
# Sampling Process
# ------------------------------------------------------------------

sample_batch_size = 100
sample_spatial_dim = 16

sampler_fn = partial(sampler,
                     spatial_dim=sample_spatial_dim,
                     marginal_prob_std=marginal_prob_std_fn,
                     diffusion_coeff=diffusion_coeff_fn,
                     batch_size=sample_batch_size,
                     num_steps=steps,
                     time_noises=time_noises,
                     device=device)

torch.cuda.synchronize()
start_time = time.time()

with torch.no_grad():
    test_vorticity_latent = AEW_model.encode(test_vorticity[:1].repeat(sample_batch_size, 1, 1))
    sample_test = sampler_fn(test_vorticity_latent, diffusion_model)
    sample_pixel = AEG_model.decode(sample_test)
torch.cuda.synchronize()
end_time = time.time()
print(f"Sampling completed in {end_time - start_time:.2f} seconds.")

rel_err_col = torch.zeros(sample_batch_size, device=device)
for i in range(100):
    rel_err_col[i] = fro_err(test_nonlinear[:1], sample_pixel[i:i+1])

plt.plot(rel_err_col.cpu().numpy())
plt.show()

with torch.no_grad():
    test_nonlinear_latent = AEG_model.encode(test_nonlinear[:1])
    test_nonlinear_pixel = AEG_model.decode(test_nonlinear_latent)
    test_vorticty_pixel = AEW_model.decode(test_vorticity_latent)

fro_latent = fro_err(test_nonlinear_latent, sample_test)
mse_latent = mse_err(test_nonlinear_latent, sample_test)
fro_sample = fro_err(test_nonlinear[:1].repeat(100, 1, 1), sample_pixel)
mse_sample = mse_err(test_nonlinear[:sample_batch_size], sample_pixel)

fro_AEG = fro_err(test_nonlinear[:sample_batch_size], test_nonlinear_pixel)
mse_AEG = mse_err(test_nonlinear[:sample_batch_size], test_nonlinear_pixel)
fro_AEW = fro_err(test_vorticity[:sample_batch_size], test_vorticty_pixel)
mse_AEW = mse_err(test_vorticity[:sample_batch_size], test_vorticty_pixel)

# ------------------------------------------------------------------
# Visualization Settings
# ------------------------------------------------------------------

### Plot and save
set_seed(13)

data1 = test_nonlinear_latent[:sample_batch_size, :, :].cpu()
data2 = sample_test.cpu()
data3 = test_nonlinear[:sample_batch_size, :, :].cpu()
data4 = sample_pixel.cpu()
data5 = np.abs(data3 - data4)

# Initialize the plot with 4 rows and 4 columns
fig, axs = plt.subplots(5, 4, figsize=(20, 25), constrained_layout=True)
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
ticks_4, tick_labels_4 = create_ticks_labels(data4.shape[1])
ticks_5, tick_labels_5 = create_ticks_labels(data5.shape[1])

# Randomly sample indices equal to the number of columns (4) for clarity
indices = [torch.randint(0, 100, (1,)).item() for _ in range(4)]

# Define color scale parameters
latent_max = 2.0
latent_min = -2.0
max_val = 1.0
min_val = -1.0
err_max = 0.3
err_min = 0
cbar_ticks_latent = np.linspace(latent_min, latent_max, 6)
cbar_ticks = np.linspace(min_val, max_val, 6)
cbar_ticks_contour = np.linspace(err_min, err_max, 6)

# Plot heatmaps and contour plots
for i, idx in enumerate(indices):
    j = i % 4  # Column index

    # --- Row 1: Truth Heatmap ---
    latent_truth = data1[idx, ...].cpu().numpy()
    sns.heatmap(
        latent_truth,
        ax=axs[0, j],
        cmap='rocket',
        cbar=(j == 3),  # Show colorbar only on the last column
        vmax=latent_max,
        vmin=latent_min,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks_latent},
        square=True
    )
    axs[0, j].set_title(r"\text{Latent Truth }" + str(j + 1))
    axs[0, j].set_xticks(ticks_1)
    axs[0, j].set_yticks(ticks_1)
    axs[0, j].set_xticklabels(tick_labels_1, rotation=0)
    axs[0, j].set_yticklabels(tick_labels_1, rotation=0)
    axs[0, j].invert_yaxis()

    # --- Row 2: Generated Heatmap ---
    latent_generated = data2[idx, ...].cpu().numpy()
    sns.heatmap(
        latent_generated,
        ax=axs[1, j],
        cmap='rocket',
        cbar=(j == 3),
        vmax=latent_max,
        vmin=latent_min,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks_latent},
        square=True
    )

    axs[1, j].set_title(r"\text{Latent Generated }" + str(j + 1))
    axs[1, j].set_xticks(ticks_2)
    axs[1, j].set_yticks(ticks_2)
    axs[1, j].set_xticklabels(tick_labels_2, rotation=0)
    axs[1, j].set_yticklabels(tick_labels_2, rotation=0)
    axs[1, j].invert_yaxis()

    # --- Row 3: Truth Heatmap ---
    truth = data3[idx, ...].cpu().numpy()
    sns.heatmap(
        truth,
        ax=axs[2, j],
        cmap='rocket',
        cbar=(j == 3),
        vmax=max_val,
        vmin=min_val,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks},
        square=True
    )

    axs[2, j].set_title(r"\text{Truth }" + str(j + 1))
    axs[2, j].set_xticks(ticks_3)
    axs[2, j].set_yticks(ticks_3)
    axs[2, j].set_xticklabels(tick_labels_3, rotation=0)
    axs[2, j].set_yticklabels(tick_labels_3, rotation=0)
    axs[2, j].invert_yaxis()

    # --- Row 4: Generated Heatmap ---
    generated = data4[idx, ...].cpu().numpy()
    sns.heatmap(
        generated,
        ax=axs[3, j],
        cmap='rocket',
        cbar=(j == 3),
        vmax=max_val,
        vmin=min_val,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks},
        square=True
    )

    axs[3, j].set_title(r"\text{Generated }" + str(j + 1))
    axs[3, j].set_xticks(ticks_4)
    axs[3, j].set_yticks(ticks_4)
    axs[3, j].set_xticklabels(tick_labels_4, rotation=0)
    axs[3, j].set_yticklabels(tick_labels_4, rotation=0)
    axs[3, j].invert_yaxis()

    # --- Row 3: Error Heatmap ---
    error = data5[idx, ...].cpu().numpy()
    ax_contour = axs[4, j]
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
        vmax=err_max,
        square=True
    )

    # Add colorbar only on the last column
    if j == 3:
        cbar_contour = fig.colorbar(
            contour,
            ax=ax_contour,
            format='%.2f'
        )

    ax_contour.set_title(r"\text{Error Contour }" + str(j + 1))
    ax_contour.set_xticks(ticks_4)
    ax_contour.set_yticks(ticks_4)
    ax_contour.set_xticklabels(tick_labels_4, rotation=0)
    ax_contour.set_yticklabels(tick_labels_4, rotation=0)

# Adjust tick parameters for all axes
for ax in axs.flat:
    ax.tick_params(axis='both', which='major', labelsize=fs)


# Adjust layout and save the plot
plt.subplots_adjust(right=0.85, hspace=0.3, wspace=0.5)
plt.show()
plt.savefig(
    'C:\\UWMadisonResearch\\Joint_LDM\\Plots\\ModelWithJoint.png',
    dpi=300,
    bbox_inches='tight'
)