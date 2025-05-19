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
from utility import get_sigmas_karras, fro_err, mse_err, set_seed, energy_spectrum
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


# ------------------------------------------------------------------
# Environment Configuration
# ------------------------------------------------------------------

### Get OneDrive Path from Environment Variables
onedrive_path = 'C:\\Users\\dongx\\OneDriveUWM'


Surrogate_file_path = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "Data", "surrogate_3050_v2.h5")
with h5py.File(Surrogate_file_path, 'r') as file:
    sol = torch.tensor(file['sol'][:], device=device)
    nonlinear = torch.tensor(file['nonlinear'][:], device=device)

sol_start = sol[..., 0:1].repeat(10, 1, 1, 1)

def navier_stokes_2d_nonlinear(a, w0, f, visc, diffusion_sampler, nonlinear_truth,
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
                nonlinear_sample = diffusion_sampler(w.repeat(1, 1, 1))
                # nonlinear_sample = torch.mean(nonlinear_sample_ensemble, dim=0, keepdim=True)
                # fro_err_mean = fro_err(nonlinear_truth[..., i], nonlinear_sample)
                # print(f"Frobenius Error: {fro_err_mean:.4e}")

                # fro_err_max = 0
                # for j in range(nonlinear_sample_ensemble.shape[0]):
                #     fro_err_max_curr = fro_err(nonlinear_truth[..., i], nonlinear_sample_ensemble[j:j+1, ...])
                #     if fro_err_max_curr > fro_err_max:
                #         fro_err_max = fro_err_max_curr
                #         nonlinear_sample = nonlinear_sample_ensemble[j:j+1, ...]
                # print(f"Frobenius Error: {fro_err_max:.4e}")
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
        if i == 0:
            sol[..., 0] = w
            sol_t[0] = t
        if (i+1) % 5000 == 0:
            j = int((i+1) / 5000)
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

diffusion_model_save = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "JointAE", "Joint_diffusion_6416_sto_v2.pth")
AEG_model_save = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "JointAE", "Joint_AE_Nonlinear_6416_sto_v2.pth")
AEW_model_save = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "JointAE", "Joint_AE_Vorticity_6416_sto_v2.pth")

AEG_model.load_state_dict(torch.load(AEG_model_save))
AEW_model.load_state_dict(torch.load(AEW_model_save))
diffusion_model.load_state_dict(torch.load(diffusion_model_save))

AEG_model.eval()
AEW_model.eval()
diffusion_model.eval()

sde_time_min = 1e-3
sde_time_max = 0.4
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

sample_batch_size = 10
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



sde_time_min = 1e-3
sde_time_max = 0.4
steps = 10

time_noises = get_sigmas_karras(steps, sde_time_min, sde_time_max, device=device)

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

sample_batch_size = 10
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

sol_corrected_pcdm, sol_t, execution_time = navier_stokes_2d_nonlinear([1, 1], sol_start
[..., 0], f, nu, diffusion_sampler=sampler_orig, nonlinear_truth = nonlinear, closure=True, delta_t=1e-3, record_steps=20000, eval_steps=5)

sol_corrected_lcdm, sol_t, execution_time = navier_stokes_2d_nonlinear([1, 1], sol_start
[..., 0], f, nu, diffusion_sampler=sampler, nonlinear_truth = nonlinear, closure=True, delta_t=1e-3, record_steps=20000, eval_steps=5)


for i in range(5):
    print(f"Time: {sol_t[-i-1]:.2f}s")
    fro_err_step = fro_err(sol_corrected_lcdm[:1, :, :, i], sol[:, :, :, (i) * 5000])
    mse_err_step = mse_err(sol_corrected_lcdm[:1, :, :, i], sol[:, :, :, (i) * 5000])
    print(f"Frobenius Error: {fro_err_step:.4e}")
    print(f"MSE Error: {mse_err_step:.4e}")


sol_nocorrected, sol_t, execution_time = navier_stokes_2d_nonlinear([1, 1], sol_start[:10,..., 0], f, nu, diffusion_sampler=None, nonlinear_truth = None, closure=False, delta_t=1e-3, record_steps=20000, eval_steps=5)


sol_corrected_pcdm[0:1, :, :, 0] = sol[0:1, :, :, 0]
sol_corrected_lcdm[0:1, :, :, 0] = sol[0:1, :, :, 0]




sol_corrected_pcdm[..., 0] = sol[..., 0]
sol_corrected_lcdm[..., 0] = sol[..., 0]

fig, axs = plt.subplots(1, 3, figsize=(42, 12), gridspec_kw={'width_ratios': [1, 1, 1]})
fs=62
# Flatten the axs array and only use the first 5 subplots
axs = axs.flatten()
steps =[0, 2, 4]
for i, ax in enumerate(axs):  # Use only the first 5 axes
    index = steps[i]

    # Compute energy spectra
    KE32 = energy_spectrum(sol[0:1, :, :, index].cpu(), 1, 1, smooth=False)
    k32, E32 = KE32['k'], KE32['E']
    KE_32_PCDM = energy_spectrum(sol_corrected_pcdm[0:1, :, :, index].cpu(), 1, 1, smooth=False)
    K_32_PCDM, E32_PCDM = KE_32_PCDM['k'], KE_32_PCDM['E']
    KE_32_LCDM = energy_spectrum(sol_corrected_lcdm[0:1, :, :, index].cpu(), 1, 1, smooth=False)
    K_32_LCDM, E32_LCDM = KE_32_LCDM['k'], KE_32_LCDM['E']

    # Plot energy spectra
    ax.loglog(k32[1:], E32[1:], label=f'Truth', linewidth=6, linestyle=":")
    ax.loglog(K_32_PCDM[1:], E32_PCDM[1:], label=f'P-CDM', linewidth=6, linestyle="--")
    ax.loglog(K_32_LCDM[1:], E32_LCDM[1:], label=f'Joint L-CDM', linewidth=6, linestyle="-.")
    ax.tick_params(axis='both', which='major', length=14, width=2)
    ax.tick_params(axis='both', which='minor', length=7, width=2)
    ax.tick_params(axis='x', which='major', pad=12)

    # Add reference line for k^(-3)
    k_ref = k32[2]   # Reference k point in the middle
    E_ref = E32[2]
    k_line = np.linspace(5, 30, 100)
    E_line = E_ref * (k_line / k_ref) ** (-3)
    ax.loglog(k_line, E_line, linestyle='solid', label=r'$k^{-3}$', color='red', linewidth=6)

    # Set plot details
    ax.set_title(f'$t$ = {sol_t[index]+30:.2f}', fontsize=fs)
    ax.set_xlim(k32[1], 10**3)  # Ensure a minimum of 1 for k
    ax.tick_params(axis='both', labelsize=fs)

    ax.set_ylim(10**-21, 10**0)
    ax.set_xlim(None, 1e3)
    ticks = [10 ** -21, 10 ** -14, 10 ** -7, 10 ** 0]

    # Define corresponding labels (the first label is for the lowest tick, etc.)
    labels = ['$10^{-21}$', '$10^{-14}$', '$10^{-7}$', '$10^{0}$']

    # Apply the tick positions and labels
    ax.set_yticks(ticks)
    # ax.set_yticklabels(labels)

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    if i == 0:
        ax.set_ylabel(f'Energy ($E$)', fontsize=fs)
    if i == 0 or i == 1 or i == 2:
        ax.set_xlabel(f'Wave number ($k$)', fontsize=fs)
    if i ==1 or i == 2:
        ax.get_yaxis().set_ticks([])

handles, labels = axs[0].get_legend_handles_labels()
lege = fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=fs,bbox_to_anchor=(0.5, 1),
                  fancybox=False, edgecolor="black")
lege.get_frame().set_linewidth(2)
plt.subplots_adjust(top=0.85)
plt.tight_layout(rect=[0, 0, 1, 0.83])
plt.savefig(
    os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "Plots", "TKE_Closure_3050.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.show()














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

k_truth = np.array([8.89e+00, 1.78e+01, 2.67e+01, 3.55e+01, 4.44e+01, 5.33e+01,
       6.22e+01, 7.11e+01, 8.00e+01, 8.89e+01, 9.77e+01, 1.07e+02,
       1.16e+02, 1.24e+02, 1.33e+02, 1.42e+02, 1.51e+02, 1.60e+02,
       1.69e+02, 1.78e+02, 1.87e+02, 1.95e+02, 2.04e+02, 2.13e+02,
       2.22e+02, 2.31e+02, 2.40e+02, 2.49e+02, 2.58e+02, 2.67e+02,
       2.75e+02, 2.84e+02])
E_truth_list = [np.array([2.75e+00, 5.75e+00, 1.41e-01, 1.20e-02, 1.33e-03, 1.11e-04,
       8.64e-06, 4.41e-07, 2.44e-08, 1.47e-09, 1.05e-10, 7.58e-12,
       4.52e-13, 2.27e-14, 3.64e-15, 4.33e-16, 5.21e-17, 2.12e-18,
       2.14e-19, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00]),
                np.array([4.15e+00, 4.68e+00, 1.50e-01, 1.35e-02, 2.63e-03, 2.86e-04,
       3.67e-05, 3.36e-06, 2.92e-07, 1.87e-08, 8.86e-10, 4.47e-11,
       1.97e-12, 1.51e-13, 7.75e-15, 5.75e-15, 3.75e-15, 1.99e-15,
       9.85e-16, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00]),
                np.array([4.53e+00, 4.31e+00, 1.29e-01, 1.28e-02, 2.53e-03, 3.07e-04,
       4.36e-05, 4.53e-06, 4.56e-07, 3.46e-08, 1.94e-09, 8.86e-11,
       3.89e-12, 2.06e-13, 1.16e-14, 5.75e-15, 3.75e-15, 1.99e-15,
       9.85e-16, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00])]

E_pcdm_list = [np.array([2.75e+00, 5.75e+00, 1.41e-01, 1.20e-02, 1.33e-03, 1.11e-04,
       8.64e-06, 4.41e-07, 2.44e-08, 1.47e-09, 1.05e-10, 7.58e-12,
       4.52e-13, 2.27e-14, 3.64e-15, 4.33e-16, 5.21e-17, 2.12e-18,
       2.14e-19, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00]),
                np.array([4.19e+00, 4.75e+00, 1.46e-01, 1.35e-02, 2.73e-03, 2.96e-04,
       3.64e-05, 3.48e-06, 3.51e-07, 3.20e-08, 6.78e-09, 2.03e-09,
       1.11e-09, 5.60e-10, 4.72e-10, 4.28e-10, 3.05e-10, 2.95e-10,
       2.81e-10, 2.67e-10, 2.51e-10, 2.30e-10, 1.92e-10, 2.11e-10,
       1.64e-10, 2.26e-10, 2.44e-10, 1.35e-10, 2.01e-10, 9.24e-11,
       1.75e-10, 1.09e-10]),
                np.array([4.72e+00, 4.42e+00, 1.22e-01, 1.32e-02, 2.57e-03, 3.28e-04,
       4.22e-05, 4.22e-06, 4.38e-07, 4.38e-08, 7.88e-09, 2.39e-09,
       1.30e-09, 6.15e-10, 4.78e-10, 3.93e-10, 3.58e-10, 2.84e-10,
       3.04e-10, 2.77e-10, 2.56e-10, 2.05e-10, 2.70e-10, 2.00e-10,
       2.42e-10, 1.88e-10, 1.84e-10, 2.22e-10, 2.62e-10, 1.34e-10,
       1.89e-10, 2.22e-10])]

E_lcdm_list = [np.array([2.75e+00, 5.75e+00, 1.41e-01, 1.20e-02, 1.33e-03, 1.11e-04,
       8.64e-06, 4.41e-07, 2.44e-08, 1.47e-09, 1.05e-10, 7.58e-12,
       4.52e-13, 2.27e-14, 3.64e-15, 4.33e-16, 5.21e-17, 2.12e-18,
       2.14e-19, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00,
       0.00e+00, 0.00e+00]),
                np.array([4.31e+00, 4.67e+00, 1.42e-01, 1.38e-02, 2.55e-03, 3.01e-04,
       3.87e-05, 3.68e-06, 3.67e-07, 5.93e-08, 3.64e-08, 1.41e-08,
       1.25e-08, 6.23e-09, 4.70e-09, 4.04e-09, 2.83e-09, 2.35e-09,
       1.11e-09, 1.41e-09, 7.20e-10, 7.05e-10, 4.11e-10, 3.78e-11,
       2.60e-11, 2.56e-11, 2.54e-11, 2.50e-11, 2.38e-11, 2.63e-11,
       2.12e-11, 1.58e-11]),
                np.array([4.82e+00, 4.31e+00, 1.20e-01, 1.49e-02, 2.69e-03, 3.69e-04,
       5.20e-05, 5.77e-06, 6.46e-07, 1.22e-07, 4.48e-08, 2.90e-08,
       1.62e-08, 1.03e-08, 7.38e-09, 5.90e-09, 3.78e-09, 3.21e-09,
       2.22e-09, 2.07e-09, 1.30e-09, 1.33e-09, 1.02e-09, 8.32e-11,
       3.55e-11, 2.33e-11, 1.89e-11, 1.18e-11, 8.40e-12, 5.24e-12,
       6.04e-12, 2.59e-12])]


fig, axs = plt.subplots(1, 3, figsize=(42, 12), gridspec_kw={'width_ratios': [1, 1, 1]})
fs=62
# Flatten the axs array and only use the first 5 subplots
axs = axs.flatten()
for i, ax in enumerate(axs):  # Use only the first 5 axes
    # Plot energy spectra
    ax.loglog(k_truth, E_truth_list[i], label=f'Ground Truth', linewidth=6, linestyle="-.")
    ax.loglog(k_truth, E_pcdm_list[i], label=f'P-CDM', linewidth=6, linestyle="--")
    ax.loglog(k_truth, E_lcdm_list[i], label=f'Joint L-CDM', linewidth=6, linestyle=":")
    ax.tick_params(axis='x', which='major', length=16, width=2, labelsize=fs)
    ax.tick_params(axis='x', which='minor', length=8, width=2, labelsize=0)
    ax.tick_params(axis='y', which='major', length=16, width=2, labelsize=fs)
    ax.tick_params(axis='y', which='minor', length=8, width=2)

    # # Add reference line for k^(-3)
    # k_ref = k_truth[2]   # Reference k point in the middle
    # E_ref = E_truth[2]
    # k_line = np.linspace(7, 50, 100)
    # E_line = E_ref * (k_line / k_ref) ** (-3)
    # ax.loglog(k_line, E_line, linestyle='solid', label=r'$k^{-3}$', color='red', linewidth=6)

    # Set plot details
    ax.set_title(f'$t$ = {sol_t[i*2]+30:.2f}', fontsize=fs)
    # ax.set_xlim(k_truth[1], 10**3)  # Ensure a minimum of 1 for k
    # ax.tick_params(axis='both', labelsize=fs)

    ax.set_ylim(10**-22, 10**2)
    ax.set_xlim(None, 3 * 1e2)
    ticks = [10 ** -22, 10 ** -15, 10 ** -8, 10 ** 2]
    #
    # # Define corresponding labels (the first label is for the lowest tick, etc.)
    # labels = ['$10^{-21}$', '$10^{-14}$', '$10^{-7}$', '$10^{0}$']

    # Apply the tick positions and labels
    # ax.set_yticks(ticks)
    # ax.set_yticklabels(labels)

    for spine in ax.spines.values():
        spine.set_linewidth(2)

    if i == 0:
        ax.set_ylabel(f'Energy ($E$)', fontsize=fs)
    if i == 0 or i == 1 or i == 2:
        ax.set_xlabel(f'Wave number ($k$)', fontsize=fs)
    if i ==1 or i == 2:
        ax.get_yaxis().set_ticks([])

handles, labels = axs[0].get_legend_handles_labels()
lege = fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=fs,bbox_to_anchor=(0.5, 1.),
                  fancybox=False, edgecolor="black")
lege.get_frame().set_linewidth(2)
plt.subplots_adjust(top=0.83)
plt.tight_layout(rect=[0, 0, 1, 0.83])

plt.savefig(
    os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "Plots", "TKE_Closure_3050.png"),
    dpi=300,
    bbox_inches='tight'
)
plt.show()









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





# Time values in seconds for the x-axis
time_values = [30, 35, 40, 45, 50]
single_pcdm = [0, 0.0314, 0.0766, 0.1192, 0.1524]
mean_pcdm = [0, 0.0192, 0.0463, 0.0652, 0.0781 ]
single_lcdm = [0, 0.0237, 0.0683, 0.0874, 0.0995 ]
mean_lcdm = [0, 0.0224, 0.0564, 0.0723, 0.0849 ]

fig, ax = plt.subplots(1, 1, figsize=(54, 36))
plt.subplots_adjust(left=0.111, right=0.889, top=0.88, bottom=0.15, wspace=0.333)
fs = 60
# MSE Plot
ax.plot(time_values, single_pcdm, marker='o', linestyle=":", markersize=10, linewidth=6, label=f"Single P-CDM")
ax.plot(time_values, mean_pcdm, marker='o', linestyle="--", markersize=10, linewidth=6, label=f"Mean P-CDM")
ax.plot(time_values, single_lcdm, marker='o', linestyle="-.", markersize=10, linewidth=6, label=f"Single Joint L-CDM")
ax.plot(time_values, mean_lcdm, marker='o', linestyle="-", markersize=10, linewidth=6, label=f"Mean Joint L-CDM")
ax.set_title(r"$D_{\text{RE}}$ \text{Comparison}", fontsize=fs, pad=16)
ax.set_xlabel(r"$t$", fontsize=fs)
ax.set_ylabel(r"$D_{\text{RE}}$", fontsize=fs)
ax.set_xticks([30, 35, 40, 45, 50])
ax.set_yticks([0.00, 0.06, 0.12, 0.18])
ax.tick_params(axis='both', which='major', labelsize=fs, width=2, length=14)
for spine in ax.spines.values():
    spine.set_linewidth(2)
# Create a shared legend at the top center, outside the axes
handles, labels = ax.get_legend_handles_labels()
lege = fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=fs,
                    bbox_to_anchor=(0.5, 1), fancybox=False, edgecolor="black")
lege.get_frame().set_linewidth(2)
plt.show()


import matplotlib.gridspec as gridspec

# Time values in seconds for the x-axis
time_values = [30, 35, 40, 45, 50]

# MSE and RMSE data for simulations

sim_vort_rmse_pcdm = [0, 4.0708e-02, 6.5192e-02, 1.0772e-01, 1.2765e-01]
sim_vort_mse_pcdm = [0, 1.7156e-03, 4.2562e-03, 1.1494e-02, 1.6236e-02]

sim_vort_rmse_pcdm_ens = [0, 1.9592e-02, 3.9259e-02, 5.3989e-02, 7.4100e-02]
sim_vort_mse_pcdm_ens = [0, 3.9879e-04, 1.5659e-03, 2.9318e-03, 5.5015e-03]


sim_vort_rmse_joint = [0, 3.5634e-02 ,4.6984e-02 ,9.7719e-02, 1.1485e-01]
sim_vort_mse_joint = [0 ,1.3266e-03 ,2.2466e-03 ,9.6691e-03, 1.3292e-02]

sim_vort_rmse_joint_ens = [0, 2.0646e-02, 4.2939e-02, 6.6262e-02, 8.3996e-02]
sim_vort_mse_joint_ens = [0 ,5.7757e-04, 1.8671e-03 ,4.8354e-03, 7.8113e-03]


# Create a figure with two subplots in a 42x12 inch figure.
fig, axs = plt.subplots(1, 2, figsize=(54, 18))
# Reserve space: left/right margins, a top margin (82% of height for axes) and a bottom margin (15%)
# wspace=0.333 controls the space between the two subplots.
plt.subplots_adjust(left=0.111, right=0.889, top=0.748, bottom=0.15, wspace=0.333)

fs = 60

# MSE Plot
ax0 = axs[0]
ax0.plot(time_values, sim_vort_mse_pcdm, marker='o', linestyle=":", markersize=10, linewidth=6, label=f"Single P-CDM")
ax0.plot(time_values, sim_vort_mse_pcdm_ens, marker='o', linestyle="--", markersize=10, linewidth=6, label=f"Ensemble P-CDM")
ax0.plot(time_values, sim_vort_mse_joint, marker='o', linestyle="-.", markersize=10, linewidth=6, label=f"Single Joint L-CDM")
ax0.plot(time_values, sim_vort_mse_joint_ens, marker='o', linestyle="-", markersize=10, linewidth=6, label=f"Ensemble Joint L-CDM")
ax0.set_title(r"$D_{\text{MSE}}$ \text{Comparison}", fontsize=fs, pad=16)
ax0.set_xlabel(r"$t$", fontsize=fs)
ax0.set_ylabel(r"$D_{\text{MSE}}$", fontsize=fs)
ax0.set_xticks([30, 35, 40, 45, 50])
ax0.set_yticks([0.000, 0.01, 0.02])
ax0.tick_params(axis='both', which='major', labelsize=fs, width=2, length=14)
for spine in ax0.spines.values():
    spine.set_linewidth(2)

# RMSE Plot
ax1 = axs[1]
ax1.plot(time_values, sim_vort_rmse_pcdm, marker='o', linestyle=":", markersize=10, linewidth=6, label=f"Single P-CDM")
ax1.plot(time_values, sim_vort_rmse_pcdm_ens, marker='o', linestyle="--", markersize=10, linewidth=6, label=f"Ensemble P-CDM")
ax1.plot(time_values, sim_vort_rmse_joint, marker='o', linestyle="-.", markersize=10, linewidth=6, label=f"Single Joint L-CDM")
ax1.plot(time_values, sim_vort_rmse_joint_ens, marker='o', linestyle="-", markersize=10, linewidth=6, label=f"Ensemble Joint L-CDM")
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
lege = fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=fs,
                  bbox_to_anchor=(0.5, 1), fancybox=False, edgecolor="black")
lege.get_frame().set_linewidth(2)

# Save the figure as a PDF ensuring nothing overlaps
plt.savefig(
    os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "Plots", "MSE_RE_Comparison.png"),
    dpi=300,
    bbox_inches='tight'
)





import matplotlib.pyplot as plt
import numpy as np
import os
onedrive_path = 'C:\\Users\\dongx\\OneDriveUWM'
# Time values
time_values = np.array([30, 35, 40, 45, 50])

# Ensemble RMSE and MSE
sim_vort_rmse_pcdm_ens = np.array([0, 1.9592e-02, 3.9259e-02, 5.3989e-02, 7.4100e-02])
sim_vort_mse_pcdm_ens = np.array([0, 3.9879e-04, 1.5659e-03, 2.9318e-03, 5.5015e-03])
sim_vort_rmse_joint_ens = np.array([0, 2.0646e-02, 4.2939e-02, 6.6262e-02, 8.3996e-02])
sim_vort_mse_joint_ens = np.array([0 ,5.7757e-04, 1.8671e-03 ,4.8354e-03, 7.8113e-03])

# Updated standard deviations
std_mse_pcdm = np.array([0.0000, 0.00013, 0.00022, 0.00032, 0.00051])
std_rmse_pcdm = np.array([0.0000, 0.0016, 0.0029, 0.0038, 0.0047])
std_mse_joint = np.array([0.0000, 0.00008, 0.00014, 0.00023, 0.00035])
std_rmse_joint = np.array([0.0000, 0.0008, 0.0016, 0.0020, 0.0027])

# Create figure
fig, axs = plt.subplots(1, 2, figsize=(54, 18))
plt.subplots_adjust(left=0.111, right=0.889, top=0.748, bottom=0.15, wspace=0.333)
fs = 60

# MSE Plot
ax0 = axs[0]
ax0.plot(time_values, sim_vort_mse_pcdm_ens, linestyle="--", marker='o', linewidth=6, markersize=10, label="Ensemble P-CDM")
ax0.fill_between(time_values,
                 sim_vort_mse_pcdm_ens - 2 * std_mse_pcdm,
                 sim_vort_mse_pcdm_ens + 2 * std_mse_pcdm,
                 color='blue', alpha=0.2)

ax0.plot(time_values, sim_vort_mse_joint_ens, linestyle="-", marker='o', linewidth=6, markersize=10, label="Ensemble Joint L-CDM")
ax0.fill_between(time_values,
                 sim_vort_mse_joint_ens - 2 * std_mse_joint,
                 sim_vort_mse_joint_ens + 2 * std_mse_joint,
                 color='orange', alpha=0.2)

ax0.set_title(r"$D_{\text{MSE}}$ \text{Comparison}", fontsize=fs, pad=16)
ax0.set_xlabel(r"$t$", fontsize=fs)
ax0.set_ylabel(r"$D_{\text{MSE}}$", fontsize=fs)
ax0.set_xticks([30, 35, 40, 45, 50])
ax0.set_yticks([0.000, 0.005, 0.010])
ax0.tick_params(axis='both', which='major', labelsize=fs, width=2, length=14)
for spine in ax0.spines.values():
    spine.set_linewidth(2)

# RMSE Plot
ax1 = axs[1]
ax1.plot(time_values, sim_vort_rmse_pcdm_ens, linestyle="--", marker='o', linewidth=6, markersize=10, label="Ensemble P-CDM")
ax1.fill_between(time_values,
                 sim_vort_rmse_pcdm_ens - 2 * std_rmse_pcdm,
                 sim_vort_rmse_pcdm_ens + 2 * std_rmse_pcdm,
                 color='blue', alpha=0.2)

ax1.plot(time_values, sim_vort_rmse_joint_ens, linestyle="-", marker='o', linewidth=6, markersize=10, label="Ensemble Joint L-CDM")
ax1.fill_between(time_values,
                 sim_vort_rmse_joint_ens - 2 * std_rmse_joint,
                 sim_vort_rmse_joint_ens + 2 * std_rmse_joint,
                 color='orange', alpha=0.2)

ax1.set_title(r"$D_{\text{RE}}$ \text{Comparison}", fontsize=fs, pad=16)
ax1.set_xlabel(r"$t$", fontsize=fs)
ax1.set_ylabel(r"$D_{\text{RE}}$", fontsize=fs)
ax1.set_xticks([30, 35, 40, 45, 50])
ax1.set_yticks([0.00, 0.05, 0.10])
ax1.tick_params(axis='both', which='major', labelsize=fs, width=2, length=14)
for spine in ax1.spines.values():
    spine.set_linewidth(2)

# Shared Legend
handles, labels = ax0.get_legend_handles_labels()
lege = fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=fs,
                  bbox_to_anchor=(0.5, 1), fancybox=False, edgecolor="black")
lege.get_frame().set_linewidth(2)

# Save
plt.savefig(
    os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "Plots", "MSE_RE_Comparison_Band.png"),
    dpi=300,
    bbox_inches='tight'
)


