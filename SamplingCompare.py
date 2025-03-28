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
onedrive_path = '/mnt/c/Users/dongx/OneDriveUWM'

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

# ### Load test dataset from HDF5 file
# test_name = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "Data", "test_diffusion_nonlinear_sto_v4.h5")
# with h5py.File(test_name, 'r') as file:
#     test_nonlinear = torch.tensor(file['test_nonlinear_64'][:2000], device=device)
#     test_vorticity = torch.tensor(file['test_vorticity_64'][:2000], device=device)
#     test_forcing = torch.tensor(file['test_forcing_64'][:], device=device)


test_name = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "Data", "test_tsne_1000.h5")
with h5py.File(test_name, 'r') as file:
    test_nonlinear = torch.tensor(file['test_nonlinear_64'][:1000], device=device)
    test_vorticity = torch.tensor(file['test_vorticity_64'][:1000], device=device)

test_vorticity_flat = test_vorticity.view(test_vorticity.shape[0], -1).cpu().numpy()
test_nonlinear_flat = test_nonlinear.view(test_nonlinear.shape[0], -1).cpu().numpy()

repeated_vorticity_flat = np.repeat(test_vorticity_flat, 1, axis=0)
joint_physical_sample = np.concatenate((physical_test_sample.view(physical_test_sample.shape[0], -1).cpu().numpy(), repeated_vorticity_flat), axis=1)

joint_joint_sample = np.concatenate((joint_test_sample.view(joint_test_sample.shape[0], -1).cpu().numpy(), repeated_vorticity_flat), axis=1)

joint_separate_sample = np.concatenate((separate_test_sample.view(separate_test_sample.shape[0], -1).cpu().numpy(), repeated_vorticity_flat), axis=1)

joint_gt = np.concatenate((test_nonlinear_flat, test_vorticity_flat), axis=1)

# joint_physical_sample =physical_test_sample.view(physical_test_sample.shape[0], -1).cpu().numpy()
#
# joint_joint_sample = joint_test_sample.view(joint_test_sample.shape[0], -1).cpu().numpy()
#
# joint_separate_sample = separate_test_sample.view(separate_test_sample.shape[0], -1).cpu().numpy()
#
# joint_gt = test_nonlinear_flat

# Combine the joint pairs and create labels.
joint_all = np.concatenate([joint_gt, joint_physical_sample, joint_joint_sample, joint_separate_sample], axis=0)
labels_joint = np.array([0] * joint_gt.shape[0] + [1] * joint_physical_sample.shape[0] +
                        [2] * joint_joint_sample.shape[0] + [3] * joint_separate_sample.shape[0])

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

scaled_joint_all = StandardScaler().fit_transform(joint_all)
tsne = TSNE(n_components=2, perplexity=40, random_state=22, learning_rate='auto', init='pca')

joint_tsne  = tsne.fit_transform(scaled_joint_all)

plt.figure(figsize=(12, 8))
plt.scatter(joint_tsne[labels_joint == 0, 0], joint_tsne[labels_joint == 0, 1],
            label='Ground Truth', alpha=1, s=10)
plt.scatter(joint_tsne[labels_joint == 1, 0], joint_tsne[labels_joint == 1, 1],
            label='Generated Physical', alpha=0.2, s=10)
plt.scatter(joint_tsne[labels_joint == 2, 0], joint_tsne[labels_joint == 2, 1],
            label='Generated Joint', alpha=0.2, s=10)
plt.scatter(joint_tsne[labels_joint == 3, 0], joint_tsne[labels_joint == 3, 1],
            label='Generated Separate', alpha=0.2,s=10)
plt.title('t-SNE Visualization of Joint (w, H) Pairs')
plt.xlabel('t-SNE Dim 1')
plt.ylabel('t-SNE Dim 2')
plt.legend()
plt.show()

plt.plot([1], [1], marker='o')
plt.show()


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_joint_all)

plt.figure(figsize=(12, 8))
plt.scatter(pca_result[labels_joint == 0, 0], pca_result[labels_joint == 0, 1],
            label='Ground Truth', alpha=1, s=10)
plt.scatter(pca_result[labels_joint == 1, 0], pca_result[labels_joint == 1, 1],
            label='Generated Physical', alpha=0.2, s=10)
plt.scatter(pca_result[labels_joint == 2, 0], pca_result[labels_joint == 2, 1],
            label='Generated Joint', alpha=0.2, s=10)
plt.scatter(pca_result[labels_joint == 3, 0], pca_result[labels_joint == 3, 1],
            label='Generated Separate', alpha=0.2,s=10)
plt.title('PCA Visualization of Joint (w, H) Pairs')
plt.xlabel('PCA Dim 1')
plt.ylabel('PCA Dim 2')
plt.legend()
plt.show()




















sigma = 30
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

modes_pcdm = 6
width_pcdm = 40
P_CDM = FNO2d_Orig(marginal_prob_std_fn, modes_pcdm, modes_pcdm, width_pcdm, padding = 0, embed_dim = 512, length = 1).to(device)

### Load pre-trained model
P_CDM.load_state_dict(torch.load(os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "OriginalDiffusion", "Convection_NoSparse_NoAE_4096_sto_v2.pth")))

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
Joint_AEG_model.load_state_dict(torch.load(os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "JointAE", "Joint_AE_Nonlinear_6416_sto_v2.pth")))
Joint_AEW_model.load_state_dict(torch.load(os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "JointAE", "Joint_AE_Vorticity_6416_sto_v2.pth")))
Joint_diffusion_model.load_state_dict(torch.load(os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "JointAE", "Joint_diffusion_6416_sto_v2.pth")))

Separate_AEG_model.load_state_dict(torch.load(os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "PretrainAE", "AE_6416_nonlinear_reg_sto_v2.pth")))
Separate_AEW_model.load_state_dict(torch.load(os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "PretrainAE", "AE_6416_vorticity_reg_sto_v2.pth")))
Separate_diffusion_model.load_state_dict(torch.load(os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "PretrainAE", "PretrainAE_Diffusion_reg_sto_v2.pth")))

### Set model to evaluation mode
Joint_AEG_model.eval()
Joint_AEW_model.eval()
Joint_diffusion_model.eval()
Separate_AEG_model.eval()
Separate_AEW_model.eval()
Separate_diffusion_model.eval()
### Set seed for reproducibility
set_seed(42)
index = 15

# ------------------------------------------------------------------
# P-CDM Sampling

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
    return mean_x

sample_spatial_dim = 64

physical_sampler = partial(sampler,
                  spatial_dim=sample_spatial_dim,
                marginal_prob_std = marginal_prob_std_fn,
                diffusion_coeff = diffusion_coeff_fn,
                batch_size = sample_batch_size,
                num_steps = sample_steps,
                time_noises = time_noises,
                device = device)

torch.cuda.synchronize()
start = time.time()
with torch.no_grad():
    #physical_test_sample = physical_sampler(test_vorticity[index:index+1].repeat(sample_batch_size, 1, 1), P_CDM)
    physical_test_sample = physical_sampler(test_vorticity.repeat(1, 1, 1), P_CDM)
torch.cuda.synchronize()
end = time.time()
print('Time elapsed: {}'.format(end - start))

rel_err_col = torch.zeros(sample_batch_size, device=device)
mse_err_col = torch.zeros(sample_batch_size, device=device)
for i in range(sample_batch_size):
    rel_err_col[i] = fro_err(test_nonlinear[index:index+1], physical_test_sample[i:i+1])
    mse_err_col[i] = mse_err(test_nonlinear[index:index+1], physical_test_sample[i:i+1])

test_sample_mean = physical_test_sample.mean(dim=0, keepdim=True)
fro_sample = fro_err(test_nonlinear[index:index+1], test_sample_mean[0:1])
mse_sample = mse_err(test_nonlinear[index:index+1], test_sample_mean[0:1])
fro_sample_nonoise = fro_err(test_nonlinear[index:index+1]-5e-5 * test_forcing[index:index+1], test_sample_mean[0:1])
mse_sample = mse_err(test_nonlinear[index:index+1], physical_test_sample[index:index+1])


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


torch.cuda.synchronize()
start_time = time.time()

with torch.no_grad():
    # test_vorticity_latent = Joint_AEW_model.encode(test_vorticity[index:index+1].repeat(sample_batch_size, 1, 1))
    test_vorticity_latent = Joint_AEW_model.encode(test_vorticity.repeat(1, 1, 1))
    sample_test = joint_sampler(test_vorticity_latent, Joint_diffusion_model)
    joint_test_sample = Joint_AEG_model.decode(sample_test)
torch.cuda.synchronize()
end_time = time.time()
print(f"Sampling completed in {end_time - start_time:.4f} seconds.")

rel_err_col = torch.zeros(sample_batch_size, device=device)
mse_err_col = torch.zeros(sample_batch_size, device=device)
for i in range(sample_batch_size):
    rel_err_col[i] = fro_err(test_nonlinear[index:index+1], joint_test_sample[i:i+1])
    mse_err_col[i] = mse_err(test_nonlinear[index:index+1], joint_test_sample[i:i+1])

sample_pixel_mean = joint_test_sample.mean(dim=0, keepdim=True)
mean_fro_err = fro_err(test_nonlinear[index:index+1], sample_pixel_mean)
mean_mse_err = mse_err(test_nonlinear[index:index+1], sample_pixel_mean)

test_nonlinear_nonoise = test_nonlinear - 5e-5* test_forcing
fro_err_noise = fro_err(test_nonlinear_nonoise, test_nonlinear)




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


torch.cuda.synchronize()
start_time = time.time()

with torch.no_grad():
    # test_vorticity_latent = Separate_AEW_model.encode(test_vorticity[index:index+1].repeat(sample_batch_size, 1, 1))
    test_vorticity_latent = Separate_AEW_model.encode(test_vorticity.repeat(1, 1, 1))
    sample_test = joint_sampler(test_vorticity_latent, Separate_diffusion_model)
    separate_test_sample = Separate_AEG_model.decode(sample_test)
torch.cuda.synchronize()
end_time = time.time()
print(f"Sampling completed in {end_time - start_time:.4f} seconds.")

rel_err_col = torch.zeros(sample_batch_size, device=device)
mse_err_col = torch.zeros(sample_batch_size, device=device)
for i in range(sample_batch_size):
    rel_err_col[i] = fro_err(test_nonlinear[index:index+1], separate_test_sample[i:i+1])
    mse_err_col[i] = mse_err(test_nonlinear[index:index+1], separate_test_sample[i:i+1])

sample_pixel_mean = separate_test_sample.mean(dim=0, keepdim=True)
mean_fro_err = fro_err(test_nonlinear[index:index+1], sample_pixel_mean)
mean_mse_err = mse_err(test_nonlinear[index:index+1], sample_pixel_mean)

test_nonlinear_nonoise = test_nonlinear - 5e-5* test_forcing
fro_err_noise = fro_err(test_nonlinear_nonoise, test_nonlinear)




import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift
import matplotlib.colors as colors


def calculate_fluctuation_spectrum(samples, domain_size=1.0):
    """
    Calculate the energy spectrum of fluctuations for a set of 2D samples
    with non-normalized wavenumbers.

    Parameters:
    ----------
    samples : ndarray
        Samples array of shape (n_samples, nx, ny)
    domain_size : float
        Physical size of the domain (default=1.0)

    Returns:
    -------
    wavenumbers : ndarray
        1D array of wavenumbers (in cycles per domain, not normalized)
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

    # Create wavenumber grid in absolute units (cycles per domain)
    # Instead of using d=1.0, we'll calculate frequencies directly
    kx = np.fft.fftfreq(nx)  # Units: cycles per nx points
    ky = np.fft.fftfreq(ny)  # Units: cycles per ny points

    # Scale to get cycles per domain
    kx = fftshift(kx * nx)  # Now in cycles per domain
    ky = fftshift(ky * ny)  # Now in cycles per domain

    kx_grid, ky_grid = np.meshgrid(kx, ky)

    # Calculate magnitude of wavenumber vector at each point
    k_grid = np.sqrt(kx_grid ** 2 + ky_grid ** 2)

    # Create bins of wavenumbers - now in cycles per domain
    # Use integer binning since we're working with discrete cycles
    k_max = int(np.ceil(np.max(k_grid)))
    k_bins = np.arange(0, k_max + 1, 1)

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

# Calculate the fluctuation spectrum for the samples
k_phy, E_phy = calculate_fluctuation_spectrum(physical_test_sample.cpu().numpy())
k_joint, E_joint = calculate_fluctuation_spectrum(joint_test_sample.cpu().numpy() )
k_separate, E_separate = calculate_fluctuation_spectrum(separate_test_sample.cpu().numpy())

# Plotting the results
plt.figure(figsize=(10, 6))
plt.loglog(k_phy, E_phy, label='P-CDM', color='blue')
plt.loglog(k_joint, E_joint, label='Joint L-CDM', color='orange')
plt.loglog(k_separate, E_separate, label='Separate L-CDM', color='green')
plt.xlabel('Wavenumber (k)')
plt.ylabel('Energy Spectrum (E(k))')
plt.title('Energy Spectrum of Fluctuations')
plt.legend()
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as colors


def visualize_samples_tsne(samples_dict, n_components=2, perplexity=30, random_state=42, figsize=(15, 9), sample_size=None):
    """
    Visualize sample distributions using t-SNE dimensionality reduction.

    Parameters:
    ----------
    samples_dict : dict
        Dictionary of sample arrays {model_name: samples_array}
        where each samples_array has shape (n_samples, nx, ny)
    n_components : int
        Number of components for t-SNE (2 or 3)
    perplexity : float
        Perplexity parameter for t-SNE
    random_state : int
        Random seed for reproducibility
    figsize : tuple
        Figure size
    sample_size : int or None
        If provided, use a random subset of this many samples per model

    Returns:
    -------
    fig : matplotlib figure
        The figure object
    tsne_results : dict
        Dictionary of t-SNE results {model_name: tsne_coordinates}
    """
    # Check for 2D or 3D visualization
    if n_components not in [2, 3]:
        raise ValueError("n_components must be 2 or 3")

    # Prepare data
    all_samples = []
    sample_model_indices = []
    model_names = list(samples_dict.keys())

    for i, (model_name, samples) in enumerate(samples_dict.items()):
        # Sample a subset if requested
        if sample_size is not None and sample_size < samples.shape[0]:
            indices = np.random.choice(samples.shape[0], sample_size, replace=False)
            model_samples = samples[indices]
        else:
            model_samples = samples

        # Flatten each sample
        flattened_samples = model_samples.reshape(model_samples.shape[0], -1)

        # Add to collection
        all_samples.append(flattened_samples)
        sample_model_indices.extend([i] * flattened_samples.shape[0])

    # Combine all samples
    all_samples = np.vstack(all_samples)
    sample_model_indices = np.array(sample_model_indices)

    # Standardize the data (important for t-SNE)
    scaler = StandardScaler()
    all_samples_scaled = scaler.fit_transform(all_samples)

    # Apply t-SNE
    print(f"Running t-SNE on {all_samples_scaled.shape[0]} samples...")
    tsne = TSNE(n_components=n_components, perplexity=perplexity,
                random_state=random_state, learning_rate='auto', init='pca')
    tsne_results_combined = tsne.fit_transform(all_samples_scaled)

    # Split results by model
    tsne_results = {}
    for i, model_name in enumerate(model_names):
        mask = (sample_model_indices == i)
        tsne_results[model_name] = tsne_results_combined[mask]

    # Create visualization
    if n_components == 2:
        fig = plot_tsne_2d(tsne_results, figsize=figsize)
    else:
        fig = plot_tsne_3d(tsne_results, figsize=figsize)

    return fig, tsne_results


def plot_tsne_2d(tsne_results, figsize=(12, 10), fs=26):
    """
    Plot 2D t-SNE results with a horizontal legend bar at the top.

    Parameters:
    ----------
    tsne_results : dict
        Dictionary of t-SNE results {model_name: tsne_coordinates}
    figsize : tuple
        Figure size (width, height)
    fs : int
        Font size for labels and title

    Returns:
    -------
    fig : matplotlib figure
        The figure object with horizontal legend at the top
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Define colors and markers for each model
    colors_list = ['#1f77b4',  '#2ca02c', '#ff7f0e', '#d62728', '#9467bd']
    markers = ['o', '^', 's', 'D', 'X']

    # Plot each model's samples
    for i, (model_name, embeddings) in enumerate(tsne_results.items()):
        color = colors_list[i % len(colors_list)]
        marker = markers[i % len(markers)]

        ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            c=color,
            marker=marker,
            alpha=0.7,
            s=50,
            label=model_name,
            edgecolors='none'
        )

    # Add labels and title
    ax.set_title('t-SNE Visualization of Modeled Distributions', fontsize=fs)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=fs)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=fs)

    # Set tick font size
    ax.tick_params(axis='both', which='major', labelsize=fs - 6)
    for spine in ax.spines.values():
        spine.set_linewidth(2)

    # Place legend at the top, outside the plot area, horizontally
    legend = ax.legend(
        fontsize=fs,  # Slightly smaller than main font size
        loc='upper center',
        bbox_to_anchor=(0.5, 1.25),
        ncol=len(tsne_results),  # One column per model
        fancybox=False,
        edgecolor='black',
    )
    legend.get_frame().set_linewidth(2)

    # Adjust limits to add some padding
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    padding = 0.1
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)
    # plt.subplots_adjust(top=0.85)
    # Adjust layout - add extra space at top for the legend
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig


def plot_tsne_3d(tsne_results, figsize=(14, 12)):
    """Plot 3D t-SNE results"""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Define colors for each model
    colors_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    markers = ['o', 's', '^', 'D', 'X']

    # Plot each model's samples
    for i, (model_name, embeddings) in enumerate(tsne_results.items()):
        color = colors_list[i % len(colors_list)]
        marker = markers[i % len(markers)]

        ax.scatter(
            embeddings[:, 0],
            embeddings[:, 1],
            embeddings[:, 2],
            c=color,
            marker=marker,
            alpha=0.7,
            s=50,
            label=model_name,
            edgecolors='none'
        )

    # Add labels and legend
    ax.set_title('3D t-SNE Visualization of Sample Distributions', fontsize=16)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14)
    ax.set_zlabel('t-SNE Dimension 3', fontsize=14)
    ax.legend(fontsize=12)

    plt.tight_layout()
    return fig


def calculate_distribution_stats(tsne_results):
    """
    Calculate statistics about the sample distributions in t-SNE space.

    Parameters:
    ----------
    tsne_results : dict
        Dictionary of t-SNE results {model_name: tsne_coordinates}

    Returns:
    -------
    stats : dict
        Dictionary of distribution statistics
    """
    stats = {}

    for model_name, embeddings in tsne_results.items():
        # Calculate center (mean)
        center = np.mean(embeddings, axis=0)

        # Calculate dispersion metrics
        distances = np.linalg.norm(embeddings - center, axis=1)

        model_stats = {
            'center': center,
            'std_dev': np.std(distances),
            'median_distance': np.median(distances),
            'max_distance': np.max(distances),
            'sample_count': embeddings.shape[0]
        }

        stats[model_name] = model_stats

    return stats


def print_distribution_stats(stats):
    """Print formatted distribution statistics"""
    print("\nSample Distribution Statistics in t-SNE Space:")
    print("-" * 60)
    print(f"{'Model':<20} {'Std Dev':<10} {'Median Dist':<12} {'Max Dist':<10} {'Count':<8}")
    print("-" * 60)

    for model_name, model_stats in stats.items():
        print(f"{model_name:<20} {model_stats['std_dev']:<10.4f} "
              f"{model_stats['median_distance']:<12.4f} "
              f"{model_stats['max_distance']:<10.4f} "
              f"{model_stats['sample_count']:<8d}")
    print("-" * 60)

    # Compare dispersion ratios
    # Use first model as reference
    reference_model = list(stats.keys())[0]
    ref_std = stats[reference_model]['std_dev']

    print(f"\nDispersion Ratios (relative to {reference_model}):")
    for model_name, model_stats in stats.items():
        if model_name != reference_model:
            ratio = model_stats['std_dev'] / ref_std
            print(f"{model_name}/{reference_model}: {ratio:.4f}")

# Example usage:
models_dict = {
    'P-CDM': physical_test_sample.cpu().numpy(),

    'L-CDM': separate_test_sample.cpu().numpy(),
    'Joint L-CDM': joint_test_sample.cpu().numpy(),
    # 'Ground Truth': (test_nonlinear[index:index+1]-5e-5 * test_forcing[index:index+1]).cpu().numpy().repeat(1000, axis=0)
}
#
# # Limit to 500 samples per model to make t-SNE faster
set_seed(42)
fig, tsne_results = visualize_samples_tsne(models_dict, n_components=2,sample_size=1000)
fig.savefig(os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "Plots", "tSNE_2D_1000_samples.png"), dpi=300, bbox_inches='tight')
#
# # Calculate and print distribution statistics
stats = calculate_distribution_stats(tsne_results)
print_distribution_stats(stats)
