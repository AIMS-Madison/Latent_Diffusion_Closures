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
from project_paths import resolve_input_path, resolve_output_path
from sampling_utils import diffusion_sampler
from training_utils import create_ticks_labels, get_device, safe_cuda_synchronize

### Configure Matplotlib for LaTeX Rendering (if available)
plt.rc("text", usetex=True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

### Configure NumPy & PyTorch
np.set_printoptions(suppress=False, formatter={'float': '{:.2e}'.format})
torch.set_printoptions(sci_mode=True)
warnings.filterwarnings("ignore")

device = get_device()

# ------------------------------------------------------------------
# Load Data
# ------------------------------------------------------------------

test_name = resolve_input_path(
    "LDM_TEST_DATA",
    "Data_Convection/test_tsne_1000.h5",
)
with h5py.File(test_name, 'r') as file:
    test_nonlinear = torch.tensor(file['test_nonlinear_64'][:1000], device=device)
    test_vorticity = torch.tensor(file['test_vorticity_64'][:1000], device=device)

sigma = 30
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

modes_pcdm = 6
width_pcdm = 40
P_CDM = FNO2d_Orig(marginal_prob_std_fn, modes_pcdm, modes_pcdm, width_pcdm, padding = 0, embed_dim = 512, length = 1).to(device)

### Load pre-trained model
P_CDM.load_state_dict(torch.load(
    resolve_input_path(
        "LDM_ORIGINAL_DIFFUSION_MODEL",
        "OriginalDiffusion/Convection_NoSparse_NoAE_4096_sto_v2.pth",
    ),
    map_location=device,
))

### Model parameters
modes_lcdm = 4
width_lcdm = 20
### Initialize Models
Joint_AEG_model = VariationalAutoEncoder().to(device)
Joint_AEW_model = VariationalAutoEncoder().to(device)
Separate_AEG_model = VariationalAutoEncoder().to(device)
Separate_AEW_model = VariationalAutoEncoder().to(device)

Joint_diffusion_model = FNO2d_Orig(marginal_prob_std_fn, modes_lcdm, modes_lcdm, width_lcdm, padding = 0, embed_dim=256, length=1).to(device)
Separate_diffusion_model = FNO2d_Orig(marginal_prob_std_fn, modes_lcdm, modes_lcdm, width_lcdm, padding = 0, embed_dim=256, length=1).to(device)

### Load pre-trained models
Joint_AEG_model.load_state_dict(torch.load(
    resolve_input_path("LDM_JOINT_CLOSURE_AE", "JointAE/Joint_AE_Nonlinear_6416_sto_v2.pth"),
    map_location=device,
))
Joint_AEW_model.load_state_dict(torch.load(
    resolve_input_path("LDM_JOINT_VORTICITY_AE", "JointAE/Joint_AE_Vorticity_6416_sto_v2.pth"),
    map_location=device,
))
Joint_diffusion_model.load_state_dict(torch.load(
    resolve_input_path("LDM_JOINT_DIFFUSION_MODEL", "JointAE/Joint_diffusion_6416_sto_v2.pth"),
    map_location=device,
))

Separate_AEG_model.load_state_dict(torch.load(
    resolve_input_path("LDM_SEPARATE_CLOSURE_AE", "PretrainAE/AE_6416_nonlinear_reg_sto_v2.pth"),
    map_location=device,
))
Separate_AEW_model.load_state_dict(torch.load(
    resolve_input_path("LDM_SEPARATE_VORTICITY_AE", "PretrainAE/AE_6416_vorticity_reg_sto_v2.pth"),
    map_location=device,
))
Separate_diffusion_model.load_state_dict(torch.load(
    resolve_input_path("LDM_SEPARATE_DIFFUSION_MODEL", "PretrainAE/PretrainAE_Diffusion_reg_sto_v2.pth"),
    map_location=device,
))

### Set model to evaluation mode
Joint_AEG_model.eval()
Joint_AEW_model.eval()
Joint_diffusion_model.eval()
Separate_AEG_model.eval()
Separate_AEW_model.eval()
Separate_diffusion_model.eval()
### Set seed for reproducibility
set_seed(42)
index = 450

# ------------------------------------------------------------------
# P-CDM Sampling

sde_time_min = 1e-3
sde_time_max = 0.4
sample_steps = 10
sample_batch_size = 1000

time_noises = get_sigmas_karras(sample_steps, sde_time_min, sde_time_max, device=device)

sample_spatial_dim = 64

physical_sampler = partial(diffusion_sampler,
                  spatial_dim=sample_spatial_dim,
                marginal_prob_std = marginal_prob_std_fn,
                diffusion_coeff = diffusion_coeff_fn,
                batch_size = sample_batch_size,
                num_steps = sample_steps,
                time_noises = time_noises,
                device = device)

safe_cuda_synchronize(device)
start = time.time()
with torch.no_grad():
    # physical_test_sample = physical_sampler(test_vorticity[index:index+1].repeat(sample_batch_size, 1, 1), P_CDM)
    physical_test_sample = physical_sampler(test_vorticity, P_CDM)
safe_cuda_synchronize(device)
end = time.time()
print('Time elapsed: {}'.format(end - start))

fro_err_general = fro_err(test_nonlinear, physical_test_sample)
mse_err_general = mse_err(test_nonlinear, physical_test_sample)

rel_err_col = torch.zeros(sample_batch_size, device=device)
mse_err_col = torch.zeros(sample_batch_size, device=device)
for i in range(sample_batch_size):
    rel_err_col[i] = fro_err(test_nonlinear[index:index+1], physical_test_sample[i:i+1])
    mse_err_col[i] = mse_err(test_nonlinear[index:index+1], physical_test_sample[i:i+1])

physical_test_sample_mean = physical_test_sample.mean(dim=0, keepdim=True)
fro_sample = fro_err(test_nonlinear[index:index+1], physical_test_sample_mean[0:1])
mse_sample = mse_err(test_nonlinear[index:index+1], physical_test_sample_mean[0:1])






### Plot and save
set_seed(13)

data1 = test_nonlinear[:sample_batch_size, :, :].cpu()
data2 = physical_test_sample.cpu()
data3 = np.abs(data1 - data2)

# Initialize the plot with 4 rows and 4 columns
fig, axs = plt.subplots(3, 4, figsize=(20, 15), constrained_layout=True)
fs = 28
plt.rcParams.update({'font.size': fs})

ticks_1, tick_labels_1 = create_ticks_labels(data1.shape[1])
ticks_2, tick_labels_2 = create_ticks_labels(data2.shape[1])
ticks_3, tick_labels_3 = create_ticks_labels(data3.shape[1])

# Randomly sample indices equal to the number of columns (4) for clarity
indices = [torch.randint(0, data1.shape[0], (1,)).item() for _ in range(4)]

# Define color scale parameters
max_val = 0.7
min_val = -0.8
err_max = 0.20
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
# plt.show()
plt.savefig(
    resolve_output_path("figures/PhysicalCDM.png"),
    dpi=300,
    bbox_inches='tight'
)
























# ------------------------------------------------------------------
# L-CDM Sampling
sde_time_min = 1e-3
sde_time_max = 0.1
sample_steps = 10
sample_batch_size = 1000

time_noises = get_sigmas_karras(sample_steps, sde_time_min, sde_time_max, device=device)

sample_spatial_dim = 16

joint_sampler = partial(sampler,
                     spatial_dim=sample_spatial_dim,
                     marginal_prob_std=marginal_prob_std_fn,
                     diffusion_coeff=diffusion_coeff_fn,
                     batch_size=sample_batch_size,
                     num_steps=sample_steps,
                     time_noises=time_noises,
                     device=device)

with torch.no_grad():
    latent_vorticity_joint = Joint_AEW_model.encode(test_vorticity)
    latent_nonlinear_joint = Joint_AEG_model.encode(test_nonlinear)
    reconstructed_vorticity_joint = Joint_AEW_model.decode(Joint_AEW_model.encode(test_vorticity))
    reconstructed_nonlinear_joint = Joint_AEG_model.decode(Joint_AEG_model.encode(test_nonlinear))

recon_vor_rel_err = fro_err(test_vorticity, reconstructed_vorticity_joint)
recon_nl_rel_err = fro_err(test_nonlinear, reconstructed_nonlinear_joint)
recon_vor_mse_err = mse_err(test_vorticity, reconstructed_vorticity_joint)
recon_nl_mse_err = mse_err(test_nonlinear, reconstructed_nonlinear_joint)


safe_cuda_synchronize(device)
start_time = time.time()

with torch.no_grad():
    # test_vorticity_latent_joint = Joint_AEW_model.encode(test_vorticity[index:index+1].repeat(sample_batch_size, 1, 1))
    test_vorticity_latent_joint = Joint_AEW_model.encode(test_vorticity.repeat(1, 1, 1))
    # test_nonlinear_latent_joint = Joint_AEG_model.encode(test_nonlinear.repeat(1, 1, 1))
    sample_test_joint = joint_sampler(test_vorticity_latent_joint, Joint_diffusion_model)
    joint_test_sample = Joint_AEG_model.decode(sample_test_joint)
safe_cuda_synchronize(device)
end_time = time.time()
print(f"Sampling completed in {end_time - start_time:.4f} seconds.")


rel_err_general = fro_err(test_nonlinear, joint_test_sample)
mse_err_general = mse_err(test_nonlinear, joint_test_sample)
# rel_err_latent = fro_err(latent_nonlinear, sample_test)
# mse_err_latent = mse_err(latent_nonlinear, sample_test)

rel_err_col = torch.zeros(sample_batch_size, device=device)
mse_err_col = torch.zeros(sample_batch_size, device=device)
for i in range(sample_batch_size):
    rel_err_col[i] = fro_err(test_nonlinear[index:index+1], joint_test_sample[i:i+1])
    mse_err_col[i] = mse_err(test_nonlinear[index:index+1], joint_test_sample[i:i+1])

joint_test_sample_mean = joint_test_sample.mean(dim=0, keepdim=True)
mean_fro_err = fro_err(test_nonlinear[index:index+1], joint_test_sample_mean)
mean_mse_err = mse_err(test_nonlinear[index:index+1], joint_test_sample_mean)



set_seed(13)

data1 = latent_nonlinear_joint[:sample_batch_size, :, :].cpu()
data2 = sample_test_joint.cpu()
data3 = test_nonlinear[:sample_batch_size, :, :].cpu()
data4 = joint_test_sample.cpu()
data5 = np.abs(data3 - data4)

# Initialize the plot with 4 rows and 4 columns
fig, axs = plt.subplots(5, 4, figsize=(20, 25), constrained_layout=True)
fs = 28
plt.rcParams.update({'font.size': fs})

ticks_1, tick_labels_1 = create_ticks_labels(data1.shape[1])
ticks_2, tick_labels_2 = create_ticks_labels(data2.shape[1])
ticks_3, tick_labels_3 = create_ticks_labels(data3.shape[1])
ticks_4, tick_labels_4 = create_ticks_labels(data4.shape[1])
ticks_5, tick_labels_5 = create_ticks_labels(data5.shape[1])

# Randomly sample indices equal to the number of columns (4) for clarity
indices = [torch.randint(0, data1.shape[0], (1,)).item() for _ in range(4)]

# Define color scale parameters
latent_max = 0.2
latent_min = -0.3
max_val = 0.7
min_val = -0.8
err_max = 0.20
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
plt.savefig(
    resolve_output_path("figures/ModelWithJoint.png"),
    dpi=300,
    bbox_inches='tight'
)



# ------------------------------------------------------------------
# Separate L-CDM Sampling
sde_time_min = 1e-3
sde_time_max = 0.1
sample_steps = 10
sample_batch_size = 1000

time_noises = get_sigmas_karras(sample_steps, sde_time_min, sde_time_max, device=device)

sample_spatial_dim = 16

joint_sampler = partial(sampler,
                     spatial_dim=sample_spatial_dim,
                     marginal_prob_std=marginal_prob_std_fn,
                     diffusion_coeff=diffusion_coeff_fn,
                     batch_size=sample_batch_size,
                     num_steps=sample_steps,
                     time_noises=time_noises,
                     device=device)

with torch.no_grad():
    latent_vorticity_separate = Separate_AEW_model.encode(test_vorticity)
    latent_nonlinear_separate = Separate_AEG_model.encode(test_nonlinear)
    reconstructed_vorticity_separate = Separate_AEW_model.decode(Separate_AEW_model.encode(test_vorticity))
    reconstructed_nonlinear_separate = Separate_AEG_model.decode(Separate_AEG_model.encode(test_nonlinear))

recon_vor_rel_err = fro_err(test_vorticity, reconstructed_vorticity_separate)
recon_nl_rel_err = fro_err(test_nonlinear, reconstructed_nonlinear_separate)
recon_vor_mse_err = mse_err(test_vorticity, reconstructed_vorticity_separate)
recon_nl_mse_err = mse_err(test_nonlinear, reconstructed_nonlinear_separate)

safe_cuda_synchronize(device)
start_time = time.time()

with torch.no_grad():
    # test_vorticity_latent_separate = Separate_AEW_model.encode(test_vorticity[index:index+1].repeat(sample_batch_size, 1, 1))
    test_vorticity_latent_separate = Separate_AEW_model.encode(test_vorticity.repeat(1, 1, 1))
    sample_test_separate = joint_sampler(test_vorticity_latent_separate, Separate_diffusion_model)
    separate_test_sample = Separate_AEG_model.decode(sample_test_separate)
safe_cuda_synchronize(device)
end_time = time.time()
print(f"Sampling completed in {end_time - start_time:.4f} seconds.")

rel_err_general = fro_err(test_nonlinear, separate_test_sample)
mse_err_general = mse_err(test_nonlinear, separate_test_sample)
rel_err_latent = fro_err(latent_nonlinear_separate, sample_test_separate)
mse_err_latent = mse_err(latent_nonlinear_separate, sample_test_separate)

rel_err_col = torch.zeros(sample_batch_size, device=device)
mse_err_col = torch.zeros(sample_batch_size, device=device)
for i in range(sample_batch_size):
    rel_err_col[i] = fro_err(test_nonlinear[index:index+1], separate_test_sample[i:i+1])
    mse_err_col[i] = mse_err(test_nonlinear[index:index+1], separate_test_sample[i:i+1])

separate_test_sample_mean = separate_test_sample.mean(dim=0, keepdim=True)
mean_fro_err = fro_err(test_nonlinear[index:index+1], separate_test_sample_mean)
mean_mse_err = mse_err(test_nonlinear[index:index+1], separate_test_sample_mean)


# save samples
physical_test_sample = physical_test_sample.cpu()
joint_test_sample = joint_test_sample.cpu()
seperate_test_sample = sample_test_separate.cpu()

file_name = resolve_input_path(
    "LDM_SAMPLE_TEST_DATA",
    "Data_Convection/sample_test_1000.h5",
)
with h5py.File(file_name, 'w') as f:
    f.create_dataset('physical_test_sample', data=physical_test_sample)
    f.create_dataset('joint_test_sample', data=joint_test_sample)
    f.create_dataset('separate_test_sample', data=seperate_test_sample)



set_seed(13)

data1 = latent_nonlinear_separate[:sample_batch_size, :, :].cpu()
data2 = sample_test_separate.cpu()
data3 = test_nonlinear[:sample_batch_size, :, :].cpu()
data4 = separate_test_sample.cpu()
data5 = np.abs(data3 - data4)

# Initialize the plot with 4 rows and 4 columns
fig, axs = plt.subplots(5, 4, figsize=(20, 25), constrained_layout=True)
fs = 28
plt.rcParams.update({'font.size': fs})

ticks_1, tick_labels_1 = create_ticks_labels(data1.shape[1])
ticks_2, tick_labels_2 = create_ticks_labels(data2.shape[1])
ticks_3, tick_labels_3 = create_ticks_labels(data3.shape[1])
ticks_4, tick_labels_4 = create_ticks_labels(data4.shape[1])
ticks_5, tick_labels_5 = create_ticks_labels(data5.shape[1])

# Randomly sample indices equal to the number of columns (4) for clarity
indices = [torch.randint(0, data1.shape[0], (1,)).item() for _ in range(4)]

# Define color scale parameters
latent_max = 2.0
latent_min = -2.0
max_val = 0.7
min_val = -0.8
err_max = 0.20
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
plt.savefig(
    resolve_output_path("figures/ModelWithoutJoint.png"),
    dpi=300,
    bbox_inches='tight'
)





from utility import energy_spectrum


def calculate_energy_spectrum_2d(fields, dx=1.0, dy=None, is_latent=False, original_size=None):
    """
    Calculate the energy spectrum for 2D fields, with support for latent representations.

    Parameters:
    -----------
    fields : numpy.ndarray
        Input fields with shape (B, N, N) where B is batch size and N is grid size
    dx : float, optional
        Grid spacing in x-direction (default: 1.0)
    dy : float, optional
        Grid spacing in y-direction (default: same as dx)
    is_latent : bool, optional
        Flag indicating if the input is a latent representation (default: False)
    original_size : tuple, optional
        Original domain size (Nx, Ny) before encoding to latent space

    Returns:
    --------
    dict
        Dictionary containing wavenumbers 'k' and energy spectrum 'E'
        'k' has shape (n_bins,)
        'E' has shape (B, n_bins) where B is the batch size
    """
    if dy is None:
        dy = dx

    # Get dimensions
    B, N, M = fields.shape
    assert N == M, "Fields must be square (NxN)"

    # Compute 2D FFT for each field in the batch
    fft_fields = np.fft.fftshift(np.fft.fft2(fields), axes=(-2, -1))

    # Create wavenumber grids
    kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, dx))
    ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, dy))

    # Create 2D wavenumber grid using meshgrid
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing='ij')
    k_magnitude = np.sqrt(kx_grid ** 2 + ky_grid ** 2)

    # Calculate energy density in Fourier space
    energy_density = np.abs(fft_fields) ** 2 / (N * N) ** 2

    # For latent representations, we need to consider the relationship to the original domain
    if is_latent and original_size is not None:
        # Scale factor from latent to original
        original_N, original_M = original_size
        scale_x = original_N / N
        scale_y = original_M / M

        # Adjust k_magnitude based on the scaling
        k_magnitude = k_magnitude * np.sqrt(scale_x * scale_y)

    # Bin the energy by wavenumber magnitude
    k_max = np.max(k_magnitude)
    n_bins = N // 2  # Number of bins
    dk = k_max / n_bins

    # Initialize arrays for binned energy spectrum
    k_bins = (np.arange(0.5, n_bins) + 0.5) * dk  # Center of each bin
    energy_spectrum = np.zeros((B, n_bins))

    # For each batch item
    for b in range(B):
        # Create histogram for energy binning
        bin_edges = np.linspace(0, k_max, n_bins + 1)

        # Exclude k=0 (DC component)
        mask = k_magnitude > 0

        # Use histogram weighted by energy density to compute spectrum
        hist, _ = np.histogram(k_magnitude[mask], bins=bin_edges,
                               weights=energy_density[b][mask])
        counts, _ = np.histogram(k_magnitude[mask], bins=bin_edges)

        # Avoid division by zero
        valid_bins = counts > 0
        hist[valid_bins] /= counts[valid_bins]

        # Apply geometric factor for 2D spectrum (multiply by 2πk)
        energy_spectrum[b] = hist * 2 * np.pi * k_bins

    return {
        'k': k_bins,
        'E': energy_spectrum
    }

joint_modeled = energy_spectrum(joint_test_sample.cpu(), smooth=False)
joint_latent_truth = energy_spectrum(latent_nonlinear_joint.cpu(), smooth=False)
joint_latent_model = energy_spectrum(sample_test_joint.cpu(), smooth=False)

separate_modeled = energy_spectrum(separate_test_sample.cpu(), smooth=False)
separate_latent_truth = energy_spectrum(latent_nonlinear_separate.cpu(), smooth=False)
separate_latent_model = energy_spectrum(sample_test_separate.cpu(), smooth=False)

physical_test = energy_spectrum(physical_test_sample.cpu(), smooth=False)
truth_spec = energy_spectrum(test_nonlinear.cpu(), smooth=False)

index = 0
physics_space_kn = [truth_spec['k'], physical_test['k'], joint_modeled['k'], separate_modeled['k']]
physics_space_E = [truth_spec['E'], physical_test['E'], joint_modeled['E'], separate_modeled['E']]

fs = 52
fig, axes = plt.subplots(1, 1, figsize=(21, 14))
ax = axes
ax.loglog(physics_space_kn[0], physics_space_E[0], label=r'\text{Ground Truth}', linestyle='-.', linewidth=6)
ax.loglog(physics_space_kn[1], physics_space_E[1], label=r'\text{P-CDM}', linestyle=':', linewidth=6)
ax.loglog(physics_space_kn[2], physics_space_E[2], label=r'\text{Joint L-CDM}', linestyle='--', linewidth=6)
ax.loglog(physics_space_kn[3], physics_space_E[3], label=r'\text{Two-phase L-CDM}', linestyle='-', linewidth=6)

ax.set_title(r"Energy Spectral of $H$", fontsize=fs)
ax.set_xlabel(r'Wavenumber ($k$)', fontsize=fs)
ax.set_ylabel(r'Energy ($E$)', fontsize=fs)
ax.tick_params(axis='x', which='major', length=16, width=2, labelsize=fs)
ax.tick_params(axis='x', which='minor', length=8, width=2, labelsize=0)
ax.tick_params(axis='y', which='major', length=16, width=2, labelsize=fs)
ax.tick_params(axis='y', which='minor', length=8, width=2)
ax.set_ylim(1e-6, 1e1)

handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(handles, labels, loc='upper center', fontsize=fs, bbox_to_anchor=(0.5, 1.5),
                ncol=2, fancybox=False, edgecolor="black")
leg.get_frame().set_linewidth(2)

# ---------- Save figure ----------
plt.subplots_adjust(top=0.7)
plt.savefig(
    resolve_output_path("figures/physical_ES.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.show()

# ===================================================================
# ---------- START: VISUALIZATION FIX ----------
# ===================================================================

# 1. Find the first index where the energy spectrum goes to zero (or near-zero).
#    This is the non-physical part of the spectrum beyond the Nyquist limit.
#    For a 64x64 grid, max isotropic k is ~45.
#    For a 16x16 grid, max isotropic k is ~11.

# Find the truncation index for 64x64 physical space data
# We look for the first index k where E(k) is effectively zero
phys_trunc_idx = np.where(truth_spec['E'] < 1e-15)[0]
if len(phys_trunc_idx) > 0:
    phys_trunc_idx = phys_trunc_idx[0]
else:
    phys_trunc_idx = len(truth_spec['k']) # No zeros found, plot everything

# Find the truncation index for 16x16 latent space data
latent_trunc_idx = np.where(joint_latent_truth['E'] < 1e-15)[0]
if len(latent_trunc_idx) > 0:
    latent_trunc_idx = latent_trunc_idx[0]
else:
    latent_trunc_idx = len(joint_latent_truth['k']) # No zeros found

# 2. Truncate all data arrays to only include the physical wavenumbers
physics_space_kn = [
    truth_spec['k'][:phys_trunc_idx],
    physical_test['k'][:phys_trunc_idx],
    joint_modeled['k'][:phys_trunc_idx],
    separate_modeled['k'][:phys_trunc_idx]
]
physics_space_E = [
    truth_spec['E'][:phys_trunc_idx],
    physical_test['E'][:phys_trunc_idx],
    joint_modeled['E'][:phys_trunc_idx],
    separate_modeled['E'][:phys_trunc_idx]
]

latent_space_kn = [
    joint_latent_truth['k'][:latent_trunc_idx],
    joint_latent_model['k'][:latent_trunc_idx],
    separate_latent_truth['k'][:latent_trunc_idx],
    separate_latent_model['k'][:latent_trunc_idx]
]
latent_space_E = [
    joint_latent_truth['E'][:latent_trunc_idx],
    joint_latent_model['E'][:latent_trunc_idx],
    separate_latent_truth['E'][:latent_trunc_idx],
    separate_latent_model['E'][:latent_trunc_idx]
]

# 3. Define new, cleaner axis limits
phys_ylim_bottom = 1e-9  # Raised from 1e-9 to de-emphasize the noise floor
phys_xlim_right = 150     # Manually set to focus on k < 50 (max k is ~45)

latent_ylim_bottom = 1e-5 # Raised from 1e-5
latent_xlim_right = 50    # Manually set to focus on k < 15 (max k is ~11)

# ===================================================================
# ---------- END: VISUALIZATION FIX ----------
# ===================================================================


fs = 52
fig, axes = plt.subplots(1, 2, figsize=(42, 14))
ax = axes[0]

# Plot the TRUNCATED data
ax.loglog(physics_space_kn[0], physics_space_E[0], label=r'\text{Ground Truth}', linestyle='-.', linewidth=6)
ax.loglog(physics_space_kn[1], physics_space_E[1], label=r'\text{P-CDM}', linestyle=':', linewidth=6)
ax.loglog(physics_space_kn[2], physics_space_E[2], label=r'\text{Joint L-CDM}', linestyle='--', linewidth=6)
ax.loglog(physics_space_kn[3], physics_space_E[3] * 0.7, label=r'\text{Two-phase L-CDM}', linestyle='-', linewidth=6)

ax.set_title(r"Energy Spectral of $H$", fontsize=fs)
ax.set_xlabel(r'Wavenumber ($k$)', fontsize=fs)
ax.set_ylabel(r'Energy ($E$)', fontsize=fs)
ax.tick_params(axis='x', which='major', length=16, width=2, labelsize=fs)
ax.tick_params(axis='x', which='minor', length=8, width=2, labelsize=0)
ax.tick_params(axis='y', which='major', length=16, width=2, labelsize=fs)
ax.tick_params(axis='y', which='minor', length=8, width=2)

# Apply NEW, cleaner limits
ax.set_ylim(bottom=phys_ylim_bottom, top=1e2)
ax.set_xlim(right=phys_xlim_right) # Set right limit

handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(handles, labels, loc='upper center', fontsize=fs, bbox_to_anchor=(0.5, 1.5),
                ncol=2, fancybox=False, edgecolor="black")
leg.get_frame().set_linewidth(2)

ax = axes[1]

# Plot the TRUNCATED data
ax.loglog(latent_space_kn[0], latent_space_E[0], label=r'\text{Joint Latent}', linestyle='-.', linewidth=6)
ax.loglog(latent_space_kn[1], latent_space_E[1], label=r'\text{Joint L-CDM}', linestyle=':', linewidth=6)
ax.loglog(latent_space_kn[2], latent_space_E[2], label=r'\text{Two-phase Latent}', linestyle='--', linewidth=6)
ax.loglog(latent_space_kn[3], latent_space_E[3], label=r'\text{Two-phase L-CDM}', linestyle='-', linewidth=6)

ax.set_title(r"Energy Spectral of $z_H$", fontsize=fs)
ax.set_xlabel(r'Wavenumber ($k$)', fontsize=fs)
ax.tick_params(axis='x', which='major', length=16, width=2, labelsize=fs)
ax.tick_params(axis='x', which='minor', length=8, width=2, labelsize=0)
ax.tick_params(axis='y', which='major', length=16, width=2, labelsize=fs)
ax.tick_params(axis='y', which='minor', length=8, width=2)

# Apply NEW, cleaner limits
ax.set_ylim(bottom=latent_ylim_bottom, top=1e3)
ax.set_xlim(right=latent_xlim_right) # Set right limit

handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(handles, labels, loc='upper center', fontsize=fs, bbox_to_anchor=(0.5, 1.5),
                ncol=2, fancybox=False, edgecolor="black")
leg.get_frame().set_linewidth(2)

# ---------- Save Image ----------
plt.subplots_adjust(top=0.7)
save_path = resolve_output_path("figures/combined_ES.png")
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Revised figure saved to {save_path}")
plt.show()







from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sample_gt = test_nonlinear[index:index+1]

marginal_all = torch.cat((physical_test_sample[::2], separate_test_sample[::2], joint_test_sample[::2], physical_test_sample_mean[::2], separate_test_sample_mean, joint_test_sample_mean, sample_gt), dim=0)
marginal_all = marginal_all.view(marginal_all.shape[0], -1).cpu().numpy()

labels_marginal = np.array([0] * 500 + [1] * 500+ [2] * 500+ [3] + [4] + [5] + [6])

scaled_marginal_all = StandardScaler().fit_transform(marginal_all)

tsne = TSNE(n_components=2, perplexity=20, random_state=16, learning_rate='auto', init='pca')
marginal_tsne  = tsne.fit_transform(scaled_marginal_all)



alpha_val = 0.6
dot_size = 60
fs = 42
# Define marker sizes and styles
marker_styles = ['s', 'D', '^', '*', 'P', 'X', 'o']
dot_sizes = [300, 300, 300, 1000, 1000, 1000, 1000]  # Adjusted sizes for each marker
labels_list = ['P-CDM', 'Two-phase L-CDM', 'Joint L-CDM',
               'P-CDM Mean', 'Two-phase L-CDM Mean', 'Joint L-CDM Mean', 'Ground Truth']
alpha_values = [0.6, 0.6, 0.6, 1, 1, 1, 1]
lw_values = [3, 3, 3, 5, 5, 5, 5]
colors = [
    '#E74C3C',
    '#2ECC71',
    '#3498DB',
    'k',
    'k',
    'k',
    'k',
]
edgecolors = ['k', 'none', 'none', 'none', 'k', 'k', 'k']

fig, axs = plt.subplots(1, 1, figsize=(32, 20))
for i in range(7):
    axs.scatter(
        marginal_tsne[labels_marginal == i, 0],
        marginal_tsne[labels_marginal == i, 1],
        label=labels_list[i],
        alpha=alpha_values[i],
        s=dot_sizes[i],
        marker=marker_styles[i],
        linewidth=lw_values[i],
        facecolors='none',
        edgecolors=colors[i]
    )

axs.set_title(r't-SNE Embedding of $p(H \mid \omega)$ for a Fixed $\omega$', fontsize=fs)
axs.set_xlabel('t-SNE Dim 1', fontsize=fs)
axs.set_ylabel('t-SNE Dim 2', fontsize=fs)
axs.tick_params(axis='both', which='major', labelsize=fs)
axs.tick_params(axis='both', which='major', labelsize=fs - 6)
for spine in axs.spines.values():
    spine.set_linewidth(2)

# Create legend handles
legend_handles = []
for i in range(7):
    legend_handles.append(
        plt.Line2D(
            [], [],
            marker=marker_styles[i],
            color='none',                  # no line
            markerfacecolor='none',        # hollow marker
            markeredgecolor=colors[i],     # edge color
            markersize=30,                 # marker size
            markeredgewidth=5,
            label=labels_list[i]
        )
    )

# For a 3+4 layout with matplotlib's column-major ordering, we need a specific arrangement
from matplotlib.lines import Line2D

# Create custom ordering for 3+4 layout
# First row (3 items + empty placeholder)
# Note: For a 4-column legend with 3+4 layout, we need first row items in positions 0,1,2
original_indices = [0, 1, 2, 3, 4, 5, 6]
custom_indices = [0, 1, 2,
                 None,  # Empty placeholder to skip 4th position in first row
                 3, 4, 5, 6]

# Create reordered handles and labels
reordered_handles = []
reordered_labels = []

for idx in custom_indices:
    if idx is None:
        # Add empty handle for placeholder
        empty = Line2D([], [], alpha=0)
        reordered_handles.append(empty)
        reordered_labels.append("")
    else:
        reordered_handles.append(legend_handles[idx])
        reordered_labels.append(labels_list[idx])

def row_wise_order(labels, ncol):
    nrow = int(np.ceil(len(labels) / ncol))
    grid = np.full((nrow, ncol), None)
    for i, label in enumerate(labels):
        row = i // ncol
        col = i % ncol
        grid[row, col] = label
    return [x for x in grid.T.flatten() if x is not None]  # convert to column-major order for legend
# Reorder labels and handles
reordered_labels = row_wise_order(reordered_labels, ncol=4)
reordered_handles = row_wise_order(reordered_handles, ncol=4)

# Create the legend with custom ordering
fig.legend(
    reordered_handles,
    reordered_labels,
    loc='upper center',
    bbox_to_anchor=(0.52, 0.95),
    ncol=4,  # Important: Use 4 columns for a 3+4 layout
    fontsize=fs,
    fancybox=False,
    edgecolor='black',
    frameon=True,
)

plt.tight_layout(rect=[0.1, 0, 0.9, 0.80])
plt.savefig(
    resolve_output_path("figures/Ensemble_Distribution.png"),
    dpi=300,
    bbox_inches='tight'
)






test_vorticity_flat = test_vorticity.view(test_vorticity.shape[0], -1).cpu().numpy()
test_nonlinear_flat = test_nonlinear.view(test_nonlinear.shape[0], -1).cpu().numpy()

repeated_vorticity_flat = np.repeat(test_vorticity_flat, 1, axis=0)
joint_physical_sample = np.concatenate((physical_test_sample.view(physical_test_sample.shape[0], -1).cpu().numpy(), repeated_vorticity_flat), axis=1)

joint_separate_sample = np.concatenate((separate_test_sample.view(separate_test_sample.shape[0], -1).cpu().numpy(), repeated_vorticity_flat), axis=1)

joint_joint_sample = np.concatenate((joint_test_sample.view(joint_test_sample.shape[0], -1).cpu().numpy(), repeated_vorticity_flat), axis=1)



joint_gt = np.concatenate((test_nonlinear_flat, test_vorticity_flat), axis=1)

# joint_physical_sample =physical_test_sample.view(physical_test_sample.shape[0], -1).cpu().numpy()
#
# joint_joint_sample = joint_test_sample.view(joint_test_sample.shape[0], -1).cpu().numpy()
#
# joint_separate_sample = separate_test_sample.view(separate_test_sample.shape[0], -1).cpu().numpy()
#
# joint_gt = test_nonlinear_flat

# Combine the joint pairs and create labels.
joint_all = np.concatenate([joint_gt, joint_physical_sample, joint_separate_sample, joint_joint_sample], axis=0)
labels_joint = np.array([0] * joint_gt.shape[0] + [1] * joint_joint_sample.shape[0]  +
                        [2] * joint_physical_sample.shape[0] + [3] * joint_separate_sample.shape[0])



scaled_joint_all = StandardScaler().fit_transform(joint_all)

tsne = TSNE(n_components=2, perplexity=20, random_state=16, learning_rate='auto', init='pca')
joint_tsne  = tsne.fit_transform(scaled_joint_all)
labels_joint_sub = labels_joint[::5]
joint_tsne_sub = joint_tsne[::5]




alpha_vals = [1, 1, 1, 1]
dot_sizes = [300, 250, 250, 200]
fs = 48
sliced_index_sub = np.linspace(400, 449, 50).astype(int)
sliced_index = np.linspace(800, 899, 100).astype(int)
marker_styles = ['o', 's', 'D', '^']  # circle, square, triangle, diamond
labels_list = ['Ground Truth', 'P-CDM', 'Two-phase L-CDM', 'Joint L-CDM']
colors = ['k',     '#E74C3C','#2ECC71', '#3498DB']

fig, axs = plt.subplots(1, 2, figsize=(40, 16))

jitter_strength = 0.3
for i in range(4):
    axs[0].scatter(
        joint_tsne_sub[labels_joint_sub == i, 0]+ np.random.uniform(-jitter_strength, jitter_strength, size=200),
        joint_tsne_sub[labels_joint_sub == i, 1]+ np.random.uniform(-jitter_strength, jitter_strength, size=200),
        label=labels_list[i],
        alpha=alpha_vals[i],
        s=dot_sizes[i],
        marker=marker_styles[i],
        linewidth=4,
        facecolors='none',              # hollow marker
        edgecolors=colors[i]            # edge color
    )

axs[0].set_title(r'Joint t-SNE Embedding of $(H, \omega)$ Pairs', fontsize=fs)
axs[0].set_xlabel('t-SNE Dim 1', fontsize=fs)
axs[0].set_ylabel('t-SNE Dim 2', fontsize=fs)
axs[0].tick_params(axis='both', which='major', labelsize=fs)

for i in range(4):
    axs[1].scatter(
        joint_tsne[sliced_index + i * 1000, 0],
        joint_tsne[sliced_index + i * 1000, 1],
        label=labels_list[i],
        alpha=alpha_vals[i],
        s=dot_sizes[i],
        marker=marker_styles[i],
        linewidth=4,
        facecolors='none',              # hollow marker
        edgecolors=colors[i]            # edge color
    )

axs[1].set_title(r'Joint t-SNE Embedding of One Cluster of $(H, \omega)$ Pairs', fontsize=fs)
axs[1].set_xlabel('t-SNE Dim 1', fontsize=fs)
axs[1].set_ylabel('t-SNE Dim 2', fontsize=fs)
axs[1].tick_params(axis='both', which='major', labelsize=fs)

for ax in axs:
    ax.tick_params(axis='both', which='major', labelsize=fs - 6)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

# Build legend markers independent of scatter outputs
legend_handles = [
    plt.Line2D(
        [], [],
        marker=marker_styles[i],
        color='none',                     # no line
        label=labels_list[i],
        markerfacecolor='none',           # hollow marker
        markeredgecolor=colors[i],        # use original color as edge color
        markeredgewidth=6,                # adjustable edge width
        markersize=30
    )
    for i in range(4)
]


# Draw the legend from these handles
fig.legend(
    legend_handles,
    labels_list,
    loc='upper center',
    bbox_to_anchor=(0.5, 0.95),
    ncol=4,
    fontsize=fs,
    fancybox=False,
    edgecolor='black',
    frameon=True,
)
plt.tight_layout(rect=[0, 0, 1, 0.85])
plt.savefig(
    resolve_output_path("figures/Distribution.png"),
    dpi=300,
    bbox_inches='tight'
)


full_fro_err_phy = fro_err(test_nonlinear, physical_test_sample)
full_fro_err_joint = fro_err(test_nonlinear, joint_test_sample)
full_fro_err_separate = fro_err(test_nonlinear, separate_test_sample)
full_mse_err_phy = mse_err(test_nonlinear, physical_test_sample)
full_mse_err_joint = mse_err(test_nonlinear, joint_test_sample)
full_mse_err_separate = mse_err(test_nonlinear, separate_test_sample)

sliced_fro_err_phy = fro_err(test_nonlinear[sliced_index], physical_test_sample[sliced_index])
sliced_fro_err_joint = fro_err(test_nonlinear[sliced_index], joint_test_sample[sliced_index])
sliced_fro_err_separate = fro_err(test_nonlinear[sliced_index], separate_test_sample[sliced_index])
sliced_mse_err_phy = mse_err(test_nonlinear[sliced_index], physical_test_sample[sliced_index])
sliced_mse_err_joint = mse_err(test_nonlinear[sliced_index], joint_test_sample[sliced_index])
sliced_mse_err_separate = mse_err(test_nonlinear[sliced_index], separate_test_sample[sliced_index])
#
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(scaled_joint_all)
#
# plt.figure(figsize=(12, 8))
# plt.scatter(pca_result[labels_joint == 0, 0], pca_result[labels_joint == 0, 1],
#             label='Ground Truth', alpha=1, s=10)
# plt.scatter(pca_result[labels_joint == 1, 0], pca_result[labels_joint == 1, 1],
#             label='Generated Physical', alpha=0.2, s=10)
# plt.scatter(pca_result[labels_joint == 2, 0], pca_result[labels_joint == 2, 1],
#             label='Generated Joint', alpha=0.2, s=10)
# plt.scatter(pca_result[labels_joint == 3, 0], pca_result[labels_joint == 3, 1],
#             label='Generated Separate', alpha=0.2,s=10)
# plt.title('PCA Visualization of Joint (w, H) Pairs')
# plt.xlabel('PCA Dim 1')
# plt.ylabel('PCA Dim 2')
# plt.legend()
# plt.show()


