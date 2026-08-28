### Standard Libraries
import os
import warnings
import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

### Load test dataset from HDF5 file
test_name = resolve_input_path(
    "LDM_LES_TEST_DATA",
    "LES_NSE/navier_stokes_LES_4096_1e-3.h5",
)
with h5py.File(test_name, 'r') as file:
    test_nonlinear = torch.tensor(file['closure_term'][::100], device=device)
    test_vorticity = torch.tensor(file['filtered_vorticity'][::100], device=device)

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
diffusion_model_save = resolve_input_path(
    "LDM_JOINT_DIFFUSION_MODEL",
    "JointAE/Joint_diffusion_LES_6416.pth",
)
AEG_model_save = resolve_input_path(
    "LDM_JOINT_CLOSURE_AE",
    "JointAE/Joint_AE_Closure_6416.pth",
)
AEW_model_save = resolve_input_path(
    "LDM_JOINT_VORTICITY_AE",
    "JointAE/Joint_AE_Vorticity_6416.pth",
)

AEG_model.load_state_dict(torch.load(AEG_model_save, map_location=device))
AEW_model.load_state_dict(torch.load(AEW_model_save, map_location=device))
diffusion_model.load_state_dict(torch.load(diffusion_model_save, map_location=device))
### Set Models to Evaluation Mode
AEG_model.eval()
AEW_model.eval()
diffusion_model.eval()

# ------------------------------------------------------------------
# Sampler Function
# ------------------------------------------------------------------

### Define time noises for the sampling process
sde_time_min = 1e-3
sde_time_max = 0.4
steps = 10
time_noises = get_sigmas_karras(steps, sde_time_min, sde_time_max, device=device)


# ------------------------------------------------------------------
# Sampling Process
# ------------------------------------------------------------------
sample_batch_size = 100
sample_spatial_dim = 16

sampler_fn = partial(diffusion_sampler,
                     spatial_dim=sample_spatial_dim,
                     marginal_prob_std=marginal_prob_std_fn,
                     diffusion_coeff=diffusion_coeff_fn,
                     batch_size=sample_batch_size,
                     num_steps=steps,
                     time_noises=time_noises,
                     device=device)


safe_cuda_synchronize(device)
start_time = time.time()

with torch.no_grad():
    test_vorticity_latent = AEW_model.encode(test_vorticity[:sample_batch_size])
    test_nonlinear_latent = AEG_model.encode(test_nonlinear[:sample_batch_size])
    sample_test = sampler_fn(test_vorticity_latent, diffusion_model)
    sample_pixel = AEG_model.decode(sample_test)
safe_cuda_synchronize(device)
end_time = time.time()
print(f"Sampling completed in {end_time - start_time:.4f} seconds.")

fro_err_pixel = fro_err(test_nonlinear[:sample_batch_size], sample_pixel)
mse_err_pixel = mse_err(test_nonlinear[:sample_batch_size], sample_pixel)
fro_err_latent = fro_err(test_nonlinear_latent[:sample_batch_size], sample_test)
mse_err_latent = mse_err(test_nonlinear_latent[:sample_batch_size], sample_test)


safe_cuda_synchronize(device)
start_time = time.time()

index = 10
with torch.no_grad():
    test_vorticity_latent = AEW_model.encode(test_vorticity[index:index+1].repeat(sample_batch_size, 1, 1))
    test_nonlinear_latent = AEG_model.encode(test_nonlinear[index:index+1].repeat(sample_batch_size, 1, 1))
    sample_test = sampler_fn(test_vorticity_latent, diffusion_model)
    sample_pixel = AEG_model.decode(sample_test)

    sample_test_mean = sample_test.mean(dim=0, keepdim=True)
    sample_pixel_mean = sample_pixel.mean(dim=0, keepdim=True)
safe_cuda_synchronize(device)
end_time = time.time()
print(f"Sampling completed in {end_time - start_time:.4f} seconds.")

sample_pixel_mean = sample_pixel.mean(dim=0, keepdim=True)
mean_fro_err = fro_err(test_nonlinear[index:index+1], sample_pixel_mean)

sample_latent_mean = sample_test.mean(dim=0, keepdim=True)
mean_fro_latent_err = fro_err(test_nonlinear_latent[index:index+1], sample_test_mean)


with torch.no_grad():
    test_nonlinear_latent = AEG_model.encode(test_nonlinear[:sample_batch_size])
    test_nonlinear_pixel = AEG_model.decode(test_nonlinear_latent)
    test_vorticity_latent = AEW_model.encode(test_vorticity[:sample_batch_size])
    test_vorticty_pixel = AEW_model.decode(test_vorticity_latent)

fro_latent = fro_err(test_nonlinear_latent, sample_test)
mse_latent = mse_err(test_nonlinear_latent, sample_test)
fro_sample = fro_err(test_nonlinear[:sample_batch_size], sample_pixel)
mse_sample = mse_err(test_nonlinear[:sample_batch_size], sample_pixel)

fro_AEG = fro_err(test_nonlinear[:sample_batch_size], test_nonlinear_pixel)
mse_AEG = mse_err(test_nonlinear[:sample_batch_size], test_nonlinear_pixel)
fro_AEW = fro_err(test_vorticity[:sample_batch_size], test_vorticty_pixel)
mse_AEW = mse_err(test_vorticity[:sample_batch_size], test_vorticty_pixel)