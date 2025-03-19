### Standard Libraries
import os
import warnings
import time

### Scientific Computing & Deep Learning Libraries
import numpy as np
import torch
import h5py
import math
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


# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device('cuda')
else:
    print("CUDA is not available.")
    device = torch.device('cpu')


### Get OneDrive Path from Environment Variables
onedrive_path = os.getenv("ONEDRIVE_PATH")
if not onedrive_path:
    raise ValueError("OneDrive path not found. Please check the environment variable!")

def navier_stokes_2d_nonlinear(a, w0, f, visc, diffusion_sampler,
                           closure = False, delta_t=1e-4, record_steps=1, eval_steps=10):
    # Grid size - must be power of 2
    N1, N2 = w0.size()[-2], w0.size()[-1]

    # Maximum frequency
    k_max = math.floor(N1 / 2.0)

    # Initial vorticity to Fourier space
    w_h = torch.fft.rfft2(w0)
    # Forcing to Fourier space
    f_h = torch.fft.rfft2(f)
    # If same forcing for the whole batch
    if len(f_h.size()) < len(w_h.size()):
        f_h = torch.unsqueeze(f_h, 0)

    # Wavenumbers in y-direction
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device),
                     torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N1, 1)
    # Wavenumbers in x-direction
    k_x = k_y.transpose(0, 1)

    # Truncate redundant modes
    k_x = k_x[..., :k_max + 1]
    k_y = k_y[..., :k_max + 1]

    # Physical wavenumbers
    kx_2d = 2.0 * torch.pi * k_x / a[0]
    ky_2d = 2.0 * torch.pi * k_y / a[1]

    # Negative Laplacian in Fourier space
    lap = kx_2d ** 2 + ky_2d ** 2
    lap[0, 0] = 1.0

    sol = torch.zeros(*w0.size(), 5, device=w0.device)
    sol_t = torch.zeros(5, device=w0.device)

    t = 0.0

    start_time = time.time()
    for i in tqdm(range(record_steps)):
        w = torch.fft.irfft2(w_h, s=(N1, N2))

        if closure == True:
            if i % eval_steps == 0:
                nonlinear_sample = diffusion_sampler(w)
            else:
                nonlinear_sample = nonlinear_sample + torch.randn_like(nonlinear_sample) * 0.00005

            # convection term
            nonlinear_h = torch.fft.rfft2(nonlinear_sample)

            w_h = ((w_h  + delta_t * f_h + delta_t * nonlinear_h
                            - 0.5 * delta_t * visc * lap * w_h)
                           / (1.0 + 0.5 * delta_t * visc * lap))

        if closure == False:
            w_h = ((w_h  + delta_t * f_h
                            - 0.5 * delta_t * visc * lap * w_h)
                           / (1.0 + 0.5 * delta_t * visc * lap))

        if (i+1) % 4000 == 0:
            j = int((i+1) / 4000) -1
            sol[..., j] = w
            sol_t[j] = t
        t += delta_t
    end_time = time.time()

    execution_time = end_time - start_time
    return sol, sol_t, execution_time

sigma = 30
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

modes = 4
width = 20
padding = 0

AEG_model = VariationalAutoEncoder().to(device)
AEW_model = VariationalAutoEncoder().to(device)
diffusion_model = FNO2d_Orig(marginal_prob_std_fn, modes, modes, width, padding, embed_dim = 256, length=1).to(device)

# AEG_model.load_state_dict(torch.load('PretrainAE\\AE_6416_nonlinear_reg_sto_v2.pth'))
# AEW_model.load_state_dict(torch.load('PretrainAE\\AE_6416_vorticity_reg_sto.pth'))
# diffusion_model.load_state_dict(torch.load('PretrainAE\\PretrainAE_Diffusion_reg_sto.pth'))

# #

### Load Pretrained Weights
diffusion_model_save = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "JointAE", "Joint_diffusion_6416_sto_v2.pth")
AEG_model_save = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "JointAE", "Joint_AE_Nonlinear_6416_sto_v2.pth")
AEW_model_save = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "JointAE", "Joint_AE_Vorticity_6416_sto_v2.pth")

AEG_model.load_state_dict(torch.load(AEG_model_save))
AEW_model.load_state_dict(torch.load(AEW_model_save))
diffusion_model.load_state_dict(torch.load(diffusion_model_save))



Surrogate_file_path = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "Data", "surrogate_3050_v2.h5")
with h5py.File(Surrogate_file_path, 'r') as file:
    sol = torch.tensor(file['sol'][:], device=device)
    nonlinear = torch.tensor(file['nonlinear'][:], device=device)

sol = sol[..., 0:1].repeat(10, 1, 1, 1)
# nonlinear = nonlinear.repeat(2000, 1, 1, 1)


AEG_model.eval()
AEW_model.eval()
diffusion_model.eval()

sde_time_min = 1e-3
sde_time_max = 0.1
steps = 10

time_noises = get_sigmas_karras(steps, sde_time_min, sde_time_max, device=device)


def sampler(vorticity_condition,
           AEG_model,
            AEW_model,
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
        encoded_vorticity_condition = AEW_model.encode(vorticity_condition)
        for i in range(num_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_noises[i]
            step_size = time_noises[i] - time_noises[i + 1]
            g = diffusion_coeff(batch_time_step)
            grad = score_model(batch_time_step, x, encoded_vorticity_condition)

            mean_x = x + (g ** 2)[:, None, None] * grad * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None] * torch.randn_like(x)
        decoded_x = AEG_model.decode(mean_x)
    return decoded_x

sample_batch_size = 1
sample_spatial_dim = 16

sampler = partial(sampler,
                    AEG_model = AEG_model,
                    AEW_model = AEW_model,
                     score_model = diffusion_model,
                  spatial_dim=sample_spatial_dim,
                marginal_prob_std = marginal_prob_std_fn,
                diffusion_coeff = diffusion_coeff_fn,
                batch_size = sample_batch_size,
                num_steps = steps,
                time_noises = time_noises,
                device = device)


modes = 6
width = 40
padding = 0
diffusion_model = FNO2d_Orig(marginal_prob_std_fn, modes, modes, width, padding, embed_dim = 512, length=1).to(device)
diffusion_model_save = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "OriginalDiffusion", "Convection_NoSparse_NoAE_4096_sto_v2.pth")
diffusion_model.load_state_dict(torch.load(diffusion_model_save))
def sampler_orig(vorticity_condition,
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

sample_batch_size = 1
sample_spatial_dim = 64

sampler_orig = partial(sampler_orig,
                     score_model = diffusion_model,
                  spatial_dim=sample_spatial_dim,
                marginal_prob_std = marginal_prob_std_fn,
                diffusion_coeff = diffusion_coeff_fn,
                batch_size = sample_batch_size,
                num_steps = steps,
                time_noises = time_noises,
                device = device)



# Viscosity parameter
nu = 1e-3

# Spatial Resolution
s = 64

# Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
t = torch.linspace(0, 1, s + 1, device=device)
t = t[0:-1]

X, Y = torch.meshgrid(t, t)
f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

set_seed(42)

sol_corrected, sol_t, execution_time = navier_stokes_2d_nonlinear([1, 1], sol[..., 0], f, nu, diffusion_sampler=sampler_orig, closure=True, delta_t=1e-3, record_steps=20000, eval_steps=5)

fro_err_last = fro_err(sol_corrected[:, :, :, -1], sol[:, :, :, -2])
for i in range(5):
    print(f"Time: {sol_t[-i-1]:.2f}s")
    fro_err_step = fro_err(sol_corrected[:, :, :, i], sol[:, :, :, (i+1) * 4000])
    print(f"Frobenius Error: {fro_err_step:.4e}")


sol_nocorrected, sol_t, execution_time = navier_stokes_2d_nonlinear([1, 1], sol[:1,..., 0], f, nu, diffusion_sampler=None,
                                                                    closure=False, delta_t=1e-3, record_steps=20000, eval_steps=5)



import matplotlib.pyplot as plt

# Data from the table
batch_sizes = [1, 10, 100, 500, 1000, 1500]
l_cdm_times = [145.49, 160.92, 315.24, 1042.21, 1830.55, 2763.87]
p_cdm_times = [180.23, 215.74, 824.15, 4225.34, 7567.64, 13820.65]
no_correction_times = [2.17, 2.22, 2.63, 2.99, 3.03, 5.59]
fs = 30

# Create figure and axes
fig, ax = plt.subplots(figsize=(21, 12))
plt.subplots_adjust(left=0.111, right=0.889, top=0.87, bottom=0.15, wspace=0.333)

# Plot data on the axes
ax.plot(batch_sizes, l_cdm_times, marker='o', linestyle=":", markersize=10, linewidth=6, label='L-CDM')
ax.plot(batch_sizes, p_cdm_times, marker='o', linestyle="--", markersize=10, linewidth=6, label='P-CDM')
ax.plot(batch_sizes, no_correction_times, marker='o', linestyle="-.", markersize=10, linewidth=6, label='No Correction')

# Set labels and title with specified fontsize
ax.set_xlabel("Batch Size (Number of Trajectories)", fontsize=fs)
ax.set_ylabel("Execution Time (seconds)", fontsize=fs)
ax.set_title("Execution Time vs. Batch Size", fontsize=fs)
plt.xticks(fontsize=fs)
plt.yticks(fontsize=fs)

# Create a shared legend at the top center, outside the axes
handles, labels = ax.get_legend_handles_labels()
lege = fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=fs,
                  bbox_to_anchor=(0.5, 1), fancybox=False, edgecolor="black")
lege.get_frame().set_linewidth(2)

# Set the axes (plot) frame linewidth to 2
for spine in ax.spines.values():
    spine.set_linewidth(2)

plt.savefig(
    'C:\\UWMadisonResearch\\Joint_LDM\\Plots\\Efficiency.png',
    dpi=300,
    bbox_inches='tight'
)




k = 0
fs = 37
# Create a figure and a grid of subplots
fig, axs = plt.subplots(5, 5, figsize=(25, 27), gridspec_kw={'width_ratios': [1]*4 + [1.073]})

# Plot each row using seaborn heatmap
for row in range(5):
    for i in range(5):  # Loop through all ten columns
        ax = axs[row, i]

        j = i * 4999
        generated = sol_corrected[k, :, :, j].cpu()
        generated_nog = sol_nocorrected[k, :, :, j].cpu()
        truth = sol[k, :, :, j+1].cpu()
        error_field = abs(generated - truth)
        error_field_nog = abs(generated_nog - truth)

        rmse = fro_err(torch.tensor(generated.unsqueeze(0)), torch.tensor(truth.unsqueeze(0)))
        mse = mse_err(torch.tensor(generated.unsqueeze(0)), torch.tensor(truth.unsqueeze(0)))
        rmse_nog = fro_err(torch.tensor(generated_nog.unsqueeze(0)), torch.tensor(truth.unsqueeze(0)))
        mse_nog = mse_err(torch.tensor(generated_nog.unsqueeze(0)), torch.tensor(truth.unsqueeze(0)))

        if row == 0:
            print(f"Time: {sol_t[j] + 30:.2f}s")
            print(f"{mse.item():.4e}")
            print(f"{rmse.item():.4e}")
            print(f"{mse_nog.item():.4e}")
            print(f"{rmse_nog.item():.4e}")


        # Set individual vmin and vmax based on the row
        if row == 0:
            data = truth
            vmin, vmax = -2.4, 2.4  # Limits for Truth and Generated rows
            ax.set_title(f't = {sol_t[j]+30:.2f}s', fontsize=fs)

            sns.heatmap(data, ax=ax, cmap="rocket", vmin=vmin, vmax=vmax, square=True, cbar=False)
        elif row == 1:
            data = generated
            vmin, vmax = -2.4, 2.4  # Limits for Truth and Generated rows

            sns.heatmap(data, ax=ax, cmap="rocket", vmin=vmin, vmax=vmax, square=True, cbar=False)
        elif row == 2:
            data = generated_nog
            vmin, vmax = -2.4, 2.4

            sns.heatmap(data, ax=ax, cmap="rocket", vmin=vmin, vmax=vmax, square=True, cbar=False)
        elif row == 3:
            data = error_field
            vmin, vmax = 0, 3.0
            cbar_ticks_contour = np.linspace(vmin, vmax, 6)
            S = data.shape[0]
            x = np.arange(S)
            y = np.arange(S)
            X, Y = np.meshgrid(x, y)
            ax_contour = ax.contourf(X, Y, data, levels=cbar_ticks_contour,
                                     cmap="rocket", vmin=vmin, vmax=vmax)
            ax.set_aspect('equal', adjustable='box')
        else:
            data = error_field_nog
            vmin, vmax = 0, 3.0
            cbar_ticks_contour = np.linspace(vmin, vmax, 6)
            S = data.shape[0]
            x = np.arange(S)
            y = np.arange(S)
            X, Y = np.meshgrid(x, y)
            ax_contour = ax.contourf(X, Y, data, levels=cbar_ticks_contour,
                                     cmap="rocket", vmin=vmin, vmax=vmax)
            ax.set_aspect('equal', adjustable='box')
        # Plot heatmap


        ax.axis('off')  # Turn off axis for cleaner look

        if i == 4:
            # Create a new axis for the colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            cb = plt.colorbar(ax.collections[0], cax=cax, ticks=np.linspace(vmin, vmax, 6))
            cax.tick_params(labelsize=fs)

            # Format tick labels based on the row
            if row < 3:  # For the first two rows
                cb.ax.set_yticklabels(['{:.1f}'.format(tick) for tick in np.linspace(vmin, vmax, 6)])
            # else:  # For the last row
            #     cb.ax.set_yticklabels(['{:.2f}'.format(tick) for tick in np.linspace(vmin, vmax, 5)])

# Add row titles on the side
row_titles = [r'Truth', r'Simulation with $\hat{H}$', r'Simulation w/o $\hat{H}$', r'Error with $\hat{H}$', r'Error w/o $\hat{H}$']

for ax, row_title in zip(axs[:, 0], row_titles):
    ax.annotate(row_title, xy=(0.1, 0.5), xytext=(-50, 0),
                xycoords='axes fraction', textcoords='offset points',
                ha='right', va='center', rotation=90, fontsize=fs)

plt.tight_layout()  # Adjust the subplots to fit into the figure area
plt.show()
plt.savefig(
    'C:\\UWMadisonResearch\\Joint_LDM\\Plots\\Closure_H.png',
    dpi=300,
    bbox_inches='tight'
)









import matplotlib.gridspec as gridspec

# Time values in seconds for the x-axis
time_values = [30, 35, 40, 45, 50]

# MSE and RMSE data for simulations

sim_vort_mse_II = [0, 8.4080e-05, 2.0721e-04, 3.9233e-04, 1.2893e-03 ]
sim_vort_rmse_II = [0, 1.0044e-02, 1.5738e-02, 2.2017e-02, 4.0336e-02 ]
sim_vort_mse_III = [0, 1.5680e-03, 1.5416e-03, 6.2521e-03, 8.2225e-03 ]
sim_vort_rmse_III = [0, 4.3053e-02, 4.2819e-02, 8.7477e-02, 1.2425e-01 ]
sim_vort_mse_IV = [0, 1.5339e-04, 6.4931e-04, 1.0665e-03, 2.6753e-03 ]
sim_vort_rmse_IV = [0, 1.3518e-02, 2.7953e-02, 3.6438e-02, 5.8637e-02 ]





# Create a figure with two subplots in a 42x12 inch figure.
fig, axs = plt.subplots(1, 2, figsize=(42, 12))
plt.subplots_adjust(left=0.111, right=0.889, top=0.748, bottom=0.15, wspace=0.333)

fs = 60

# MSE Plot
ax0 = axs[0]
ax0.plot(time_values, sim_vort_mse_II, marker='o', linestyle=":", markersize=10, linewidth=6, label=f"Simulation II")
ax0.plot(time_values, sim_vort_mse_III, marker='o', linestyle="--", markersize=10, linewidth=6, label=f"Simulation III")
ax0.plot(time_values, sim_vort_mse_IV, marker='o', linestyle="-.", markersize=10, linewidth=6, label=f"Simulation IV")
ax0.set_title(r"$D_{\text{MSE}}$ \text{Comparison}", fontsize=fs, pad=16)
ax0.set_xlabel(r"$t$", fontsize=fs)
ax0.set_ylabel(r"$D_{\text{MSE}}$", fontsize=fs)
ax0.set_xticks([30, 35, 40, 45, 50])
ax0.set_yticks([0.000, 0.003, 0.006, 0.009])
ax0.tick_params(axis='both', which='major', labelsize=fs, width=2, length=14)
for spine in ax0.spines.values():
    spine.set_linewidth(2)

# RMSE Plot
ax1 = axs[1]
ax1.plot(time_values, sim_vort_rmse_II, marker='o', linestyle=":", markersize=10, linewidth=6, label=f"Simulation II")
ax1.plot(time_values, sim_vort_rmse_III, marker='o', linestyle="--", markersize=10, linewidth=6, label=f"Simulation III")
ax1.plot(time_values, sim_vort_rmse_IV, marker='o', linestyle="-.", markersize=10, linewidth=6, label=f"Simulation IV")
ax1.set_title(r"$D_{\text{RE}}$ \text{Comparison}", fontsize=fs, pad=16)
ax1.set_xlabel(r"$t$", fontsize=fs)
ax1.set_ylabel(r"$D_{\text{RE}}$", fontsize=fs)
ax1.set_xticks([30, 35, 40, 45, 50])
ax1.set_yticks([0.00, 0.05, 0.10, 0.15])
ax1.tick_params(axis='both', which='major', labelsize=fs, width=2, length=14)
for spine in ax1.spines.values():
    spine.set_linewidth(2)

# Create a shared legend at the top center, outside the axes
handles, labels = ax0.get_legend_handles_labels()
lege = fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=fs,
                  bbox_to_anchor=(0.5, 1), fancybox=False, edgecolor="black")
lege.get_frame().set_linewidth(2)

# Save the figure as a PDF ensuring nothing overlaps
plt.savefig('C:\\UWMadisonResearch\\Joint_LDM\\Plots\\MSE_RE_Comparison_H.png', dpi=300)
plt.show()