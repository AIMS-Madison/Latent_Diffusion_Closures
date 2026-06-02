# Cell 1: Imports and Setup
import torch
import torch.nn.functional as F
import numpy as np
import math
from tqdm.notebook import tqdm
from scipy.stats import binned_statistic


# Helper Class and Kernel Definition
def get_velocity_from_vorticity(w_field, domain_size):
    """
    Calculates the u and v velocity fields from a 2D vorticity field.

    Args:
        w_field (np.ndarray): A 2D (N, N) vorticity field.
        domain_size (float): The physical size of the domain.

    Returns:
        tuple[np.ndarray, np.ndarray]: The u and v velocity fields.
    """
    N = w_field.shape[0]

    # Go to Fourier space
    w_h = np.fft.fft2(w_field)

    # Create the wavenumber grid
    k_vec = (2 * np.pi / domain_size) * np.fft.fftfreq(N, d=1.0/N)
    kx, ky = np.meshgrid(k_vec, k_vec)
    k_sq = kx**2 + ky**2

    # Avoid division by zero at the k=0 mode
    inv_k_sq = np.divide(1.0, k_sq, out=np.zeros_like(k_sq), where=k_sq!=0)

    # Calculate streamfunction in Fourier space
    psi_h = -w_h * inv_k_sq

    # Calculate velocity components in Fourier space
    u_h = 1j * ky * psi_h
    v_h = -1j * kx * psi_h

    # Go back to physical space
    u = np.fft.ifft2(u_h).real
    v = np.fft.ifft2(v_h).real

    return u, v

# --- 2. New Function to Calculate TKE Spectrum ---

def get_TKE_spectrum(u, v, domain_size):
    """
    [REVISED] Calculates the 1D radially-averaged TKE spectrum and
    removes any NaN values that result from empty wavenumber bins.
    """
    N = u.shape[0]

    # Go to Fourier space
    u_h = np.fft.fft2(u)
    v_h = np.fft.fft2(v)

    # Calculate 2D TKE spectral density
    tke_2d = 0.5 * (np.abs(u_h)**2 + np.abs(v_h)**2) / (N**2)

    # Create the wavenumber grid
    k_vec = (2 * np.pi / domain_size) * np.fft.fftfreq(N, d=1.0/N)
    kx, ky = np.meshgrid(k_vec, k_vec)
    k_mag = np.sqrt(kx**2 + ky**2)

    # Define radial bins for averaging
    k_bins = np.arange(0.5, N//2, 1.)

    # Average the 2D spectrum into 1D bins
    tke_1d_raw, _, _ = binned_statistic(
        k_mag.flatten(),
        tke_2d.flatten(),
        statistic='mean',
        bins=k_bins
    )

    k_vals_raw = 0.5 * (k_bins[:-1] + k_bins[1:]) * 2 * np.pi

    # --- [CRITICAL FIX] ---
    # Find the indices where the binned means are NOT NaN
    valid_indices = ~np.isnan(tke_1d_raw)

    # Select only the valid data points
    k_vals_clean = k_vals_raw[valid_indices]
    tke_1d_clean = tke_1d_raw[valid_indices]

    return k_vals_clean, tke_1d_clean

class GaussianRF:
    """
    Generates Gaussian Random Fields for a specified domain size.
    """
    def __init__(self, size, domain_size=1, alpha=2.5, tau=7, device=None):
        self.device = device
        k_max = size // 2

        # 1. Create integer wavenumbers (as before)
        wavenumers_int = torch.cat((torch.arange(0, k_max), torch.arange(-k_max, 0)), 0).to(device).repeat(size, 1)
        k_x_int = wavenumers_int.transpose(0, 1)
        k_y_int = wavenumers_int

        # 2. [NEW] Scale them to get physical wavenumbers
        scaling_factor = 2.0 * math.pi / domain_size
        k_x = k_x_int * scaling_factor
        k_y = k_y_int * scaling_factor

        sigma = tau**(0.5*(2*alpha - 2))

        # 3. Use physical wavenumbers in the spectrum equation
        self.sqrt_eig = (size**2) * math.sqrt(2.0) * sigma * ((k_x**2 + k_y**2) + tau**2)**(-alpha / 2.0)
        self.sqrt_eig[0, 0] = 0.0
        self.size = tuple([size, size])

    def sample(self, N):
        coeff = torch.randn(N, *self.size, 2, device=self.device)
        coeff[..., 0] = self.sqrt_eig * coeff[..., 0]
        coeff[..., 1] = self.sqrt_eig * coeff[..., 1]
        u = torch.fft.ifftn(torch.view_as_complex(coeff), dim=[1, 2]).real
        return u

def define_fourier_gaussian_kernel(N, domain_size, delta_frac, device='cpu'):
    """
    Defines a Gaussian filter kernel directly in 2D Fourier (rfft2) space.

    Args:
        N (int): The grid resolution (N x N).
        domain_size (float): The physical size of the domain (e.g., 1.0 or 2*pi).
        delta_frac (float): The filter width as a multiple of the grid spacing.
        device (str): The torch device ('cpu' or 'cuda').

    Returns:
        torch.Tensor: The filter kernel in rfft2 space, ready for multiplication.
    """
    # The filter width in physical units
    delta = delta_frac * (domain_size / N)

    # Create wavenumber grid matching the rfft2 output shape
    k_max = N // 2
    k_y_int = torch.cat((torch.arange(0, k_max), torch.arange(-k_max, 0)), 0).to(device)
    k_x_int = k_y_int.view(N, 1)

    # Scale to physical wavenumbers
    scaling_factor = 2.0 * math.pi / domain_size
    k_y = k_y_int * scaling_factor
    k_x = k_x_int * scaling_factor

    k_x_rfft, k_y_rfft = k_x[..., :k_max+1], k_y.view(1, N)[..., :k_max+1]

    # Calculate the squared magnitude of the wavevector
    k_sq = k_x_rfft**2 + k_y_rfft**2

    # Define the Gaussian filter in Fourier space. The factor of 24 in the
    # denominator is a standard convention in LES literature.
    fourier_kernel_h = torch.exp(-delta**2 * k_sq / 24.0)

    # Add a batch dimension to match the solver's tensor shapes
    return fourier_kernel_h.unsqueeze(0)

def define_sharp_spectral_kernel(N_hr, domain_size, downsample_factor, device='cpu'):
    """
    Defines a sharp spectral (ideal low-pass) filter in rfft space.
    This kernel is 1 for modes below the coarse-grid Nyquist frequency and 0 otherwise.

    Args:
        N_hr (int): The high-resolution grid size.
        domain_size (float): The physical size of the domain.
        downsample_factor (int): The factor by which the grid is coarsened.

    Returns:
        torch.Tensor: The sharp spectral kernel in rfft2 space.
    """
    N_cr = N_hr // downsample_factor

    # The cutoff wavenumber is the Nyquist frequency of the coarse grid
    k_max_cr = N_cr // 2
    k_cutoff_phys = k_max_cr * (2 * math.pi / domain_size)

    # Create the high-resolution wavenumber grid in rfft space
    ky_int = torch.fft.fftfreq(N_hr, d=1/N_hr).to(device)
    kx_int = torch.fft.rfftfreq(N_hr, d=1/N_hr).to(device)
    kx_grid_int, ky_grid_int = torch.meshgrid(ky_int, kx_int, indexing='ij')

    # Convert to physical wavenumbers
    scaling_factor = 2 * math.pi / domain_size
    k_mag_phys = torch.sqrt((kx_grid_int * scaling_factor)**2 + (ky_grid_int * scaling_factor)**2)

    # The kernel is 1 where k is less than or equal to the cutoff, and 0 otherwise
    sharp_kernel_h = (k_mag_phys <= k_cutoff_phys).float()

    return sharp_kernel_h.unsqueeze(0) # Add batch dimension

# --- New Helper Function for Spectral Truncation ---
def spectral_truncate(field_h_hr, N_cr):
    """
    Truncates a high-resolution field in rfft space to a low-resolution one.
    This is the "coarse-graining" step from the paper.

    Args:
        field_h_hr (torch.Tensor): The high-res field in rfft2 space (B, N_hr, N_hr/2+1).
        N_cr (int): The target coarse-grid resolution.

    Returns:
        torch.Tensor: The truncated low-res field in rfft2 space (B, N_cr, N_cr/2+1).
    """
    N_hr = field_h_hr.shape[-2]
    k_max_cr = N_cr // 2

    # Create a new tensor for the coarse-grained data
    field_h_cr = torch.zeros(
        field_h_hr.shape[0], N_cr, k_max_cr + 1,
        dtype=field_h_hr.dtype, device=field_h_hr.device
    )

    # Copy the low-frequency modes for ky >= 0
    field_h_cr[..., :k_max_cr + 1, :k_max_cr + 1] = field_h_hr[..., :k_max_cr + 1, :k_max_cr + 1]

    # Copy the low-frequency modes for ky < 0 (handles the wrap-around frequencies)
    field_h_cr[..., -k_max_cr:, :k_max_cr + 1] = field_h_hr[..., N_hr - k_max_cr:, :k_max_cr + 1]

    # The FFT normalization factor needs to be adjusted for the change in grid size
    return field_h_cr * (N_cr**2 / N_hr**2)

def solve_les_diagnostics(w0_hr, f_hr, visc, r, T, delta_t, record_steps, fourier_kernel_h, domain_size, downsample_factor=1):
    """
    [REVISED] Includes a linear friction term '-rω' in the governing equation.
    """
    L = domain_size
    N_hr = w0_hr.size()[-1]
    if N_hr % downsample_factor != 0: raise ValueError("Grid size N must be divisible by downsample_factor.")
    N_cr = N_hr // downsample_factor

    if len(w0_hr.shape) == 2: w0_hr = w0_hr.unsqueeze(0)
    if len(f_hr.shape) == 2: f_hr = f_hr.unsqueeze(0)

    # --- High-Resolution Wavenumbers ---
    k_max_hr = N_hr // 2
    k_y_int_hr = torch.cat((torch.arange(0, k_max_hr), torch.arange(-k_max_hr, 0)), 0).to(w0_hr.device)
    k_x_int_hr = k_y_int_hr.view(N_hr, 1)
    k_y_hr, k_x_hr = k_y_int_hr * (2*math.pi/L), k_x_int_hr * (2*math.pi/L)
    k_x_rfft_hr, k_y_rfft_hr = k_x_hr[..., :k_max_hr+1], k_y_hr.view(1,N_hr)[..., :k_max_hr+1]
    lap_hr = k_x_rfft_hr**2 + k_y_rfft_hr**2
    lap_hr[0, 0] = 1.0
    dealias_hr = ((torch.abs(k_x_rfft_hr) <= (2/3)*k_max_hr*(2*math.pi/L)) & (torch.abs(k_y_rfft_hr) <= (2/3)*k_max_hr*(2*math.pi/L))).unsqueeze(0)

    # --- Coarse-Resolution Wavenumbers ---
    k_max_cr = N_cr // 2
    k_y_cr_int = torch.cat((torch.arange(0, k_max_cr), torch.arange(-k_max_cr, 0)), 0).to(w0_hr.device)
    k_x_cr_int = k_y_cr_int.view(N_cr, 1)
    k_y_cr, k_x_cr = k_y_cr_int * (2*math.pi/L), k_x_cr_int * (2*math.pi/L)
    k_x_rfft_cr, k_y_rfft_cr = k_x_cr[..., :k_max_cr+1], k_y_cr.view(1,N_cr)[..., :k_max_cr+1]
    lap_cr = k_x_rfft_cr**2 + k_y_rfft_cr**2
    lap_cr[0, 0] = 1.0
    dealias_cr = ((torch.abs(k_x_rfft_cr) <= (2/3)*k_max_cr*(2*math.pi/L)) & (torch.abs(k_y_rfft_cr) <= (2/3)*k_max_cr*(2*math.pi/L))).unsqueeze(0)

    w_h, f_h = torch.fft.rfft2(w0_hr), torch.fft.rfft2(f_hr)
    steps, record_time = math.ceil(T/delta_t), math.floor(math.ceil(T/delta_t)/record_steps)
    sol_history_ds = torch.zeros(w0_hr.shape[0], N_cr, N_cr, record_steps, device=w0_hr.device)
    closure_history_ds = torch.zeros_like(sol_history_ds)

    c = 0
    for j in tqdm(range(steps), desc="Running DNS for LES Diagnostics"):
        psi_h = w_h / lap_hr
        u_x_phys = torch.fft.irfft2(1j * k_y_rfft_hr * psi_h, s=(N_hr, N_hr))
        u_y_phys = torch.fft.irfft2(-1j * k_x_rfft_hr * psi_h, s=(N_hr, N_hr))
        w_x_phys = torch.fft.irfft2(1j * k_x_rfft_hr * w_h, s=(N_hr, N_hr))
        w_y_phys = torch.fft.irfft2(1j * k_y_rfft_hr * w_h, s=(N_hr, N_hr))
        N_phys = u_x_phys * w_x_phys + u_y_phys * w_y_phys
        N_h = dealias_hr * torch.fft.rfft2(N_phys)

        # [CHANGE] The time-stepping scheme now includes the linear friction term 'r'
        denominator = 1.0 + 0.5 * delta_t * (visc * lap_hr + r)
        numerator = (1.0 - 0.5 * delta_t * (visc * lap_hr + r)) * w_h - delta_t * N_h + delta_t * f_h
        w_h = numerator / denominator

        if torch.isnan(w_h).any():
            sol_history_ds[..., c:] = float('nan')
            closure_history_ds[..., c:] = float('nan')
            break

        if j % record_time == 0 and c < record_steps:
            w_bar_h_hr = w_h * fourier_kernel_h
            N_bar_h_hr = N_h * fourier_kernel_h
            w_h_bar_cr = spectral_truncate(w_bar_h_hr, N_cr)
            N_h_bar_cr_clean = spectral_truncate(N_bar_h_hr, N_cr)

            psi_h_bar_cr = w_h_bar_cr / lap_cr
            u_x_bar_phys = torch.fft.irfft2(1j * k_y_rfft_cr * psi_h_bar_cr, s=(N_cr, N_cr))
            u_y_bar_phys = torch.fft.irfft2(-1j * k_x_rfft_cr * psi_h_bar_cr, s=(N_cr, N_cr))
            w_x_bar_phys = torch.fft.irfft2(1j * k_x_rfft_cr * w_h_bar_cr, s=(N_cr, N_cr))
            w_y_bar_phys = torch.fft.irfft2(1j * k_y_rfft_cr * w_h_bar_cr, s=(N_cr, N_cr))
            N_of_w_bar_phys = u_x_bar_phys * w_x_bar_phys + u_y_bar_phys * w_y_bar_phys

            N_h_of_w_bar_cr_aliased = torch.fft.rfft2(N_of_w_bar_phys)
            N_h_of_w_bar_cr_clean = dealias_cr * N_h_of_w_bar_cr_aliased

            Pi_h_cr = N_h_bar_cr_clean - N_h_of_w_bar_cr_clean
            Pi_phys = torch.fft.irfft2(Pi_h_cr, s=(N_cr, N_cr))

            sol_history_ds[..., c] = torch.fft.irfft2(w_h_bar_cr, s=(N_cr, N_cr))
            closure_history_ds[..., c] = Pi_phys
            c += 1

    return sol_history_ds.permute(0, 3, 1, 2), closure_history_ds.permute(0, 3, 1, 2)

def solve_deterministic_simulation(w0, f, visc, r, T, delta_t, record_steps, domain_size):
    """
    [REVISED] Runs a standard deterministic simulation and saves snapshots
    at the exact indices specified by snapshot_indices.
    """
    L = domain_size
    N = w0.size()[-1]
    device = w0.device

    if len(w0.shape) == 2: w0 = w0.unsqueeze(0)
    if len(f.shape) == 2: f = f.unsqueeze(0)

    # --- Wavenumber and Operator Setup ---
    k_max = N // 2
    k_y_int = torch.cat((torch.arange(0, k_max), torch.arange(-k_max, 0)), 0).to(device)
    k_x_int = k_y_int.view(N, 1)
    k_y, k_x = k_y_int * (2*math.pi/L), k_x_int * (2*math.pi/L)
    k_x_rfft, k_y_rfft = k_x[..., :k_max+1], k_y.view(1,N)[..., :k_max+1]
    lap = k_x_rfft**2 + k_y_rfft**2
    lap[0, 0] = 1.0
    dealias = ((torch.abs(k_x_rfft) <= (2/3)*k_max*(2*math.pi/L)) & (torch.abs(k_y_rfft) <= (2/3)*k_max*(2*math.pi/L))).unsqueeze(0)

    w_h, f_h = torch.fft.rfft2(w0), torch.fft.rfft2(f)
    steps = int(T/delta_t)

    steps, record_time = math.ceil(T/delta_t), math.floor(math.ceil(T/delta_t)/record_steps)
    sol_history = torch.zeros(w0.shape[0], N, N, record_steps, device=w0.device)

    c_record = 0

    desc = f"Running {N}x{N} Deterministic Sim"
    for j in tqdm(range(steps), desc=desc):
        # --- Standard Time-Stepping Logic ---
        psi_h = w_h / lap
        u_x_phys = torch.fft.irfft2(1j * k_y_rfft * psi_h, s=(N, N))
        u_y_phys = torch.fft.irfft2(-1j * k_x_rfft * psi_h, s=(N, N))
        w_x_phys = torch.fft.irfft2(1j * k_x_rfft * w_h, s=(N, N))
        w_y_phys = torch.fft.irfft2(1j * k_y_rfft * w_h, s=(N, N))
        N_h_unaliased = dealias * torch.fft.rfft2(u_x_phys * w_x_phys + u_y_phys * w_y_phys)

        denominator = 1.0 + 0.5 * delta_t * (visc * lap + r)
        numerator = (1.0 - 0.5 * delta_t * (visc * lap + r)) * w_h - delta_t * N_h_unaliased + delta_t * f_h
        w_h = numerator / denominator

        if torch.isnan(w_h).any():
            print(f"\nNaN detected in solution at step {j}. Ending simulation early.")
            sol_history[..., c_record:] = float('nan')
            break

        # [CHANGE] Save the state if the current step is in our target indices
        if j % record_time == 0 and c_record < record_steps:
            sol_history[..., c_record] = torch.fft.irfft2(w_h, s=(N, N))
            c_record += 1

    return sol_history.permute(0, 3, 1, 2)

def solve_coarse_with_closure(w0_cr, f_cr, visc, r, T, delta_t, record_steps, closure_history_cr, domain_size):
    """ [REVISED] Includes a linear friction term '-rω' in the governing equation. """
    L = domain_size
    N_cr = w0_cr.size()[-1]
    if len(w0_cr.shape) == 2: w0_cr = w0_cr.unsqueeze(0)
    if len(f_cr.shape) == 2: f_cr = f_cr.unsqueeze(0)

    k_max = N_cr // 2
    k_y_int = torch.cat((torch.arange(0,k_max),torch.arange(-k_max,0)),0).to(w0_cr.device)
    k_x_int = k_y_int.view(N_cr, 1)
    k_y, k_x = k_y_int * (2*math.pi/L), k_x_int * (2*math.pi/L)
    k_x_rfft, k_y_rfft = k_x[..., :k_max+1], k_y.view(1,N_cr)[..., :k_max+1]
    lap = k_x_rfft**2+k_y_rfft**2
    lap[0,0] = 1.0
    dealias = ((torch.abs(k_x_rfft) <= (2/3)*k_max*(2*math.pi/L)) & (torch.abs(k_y_rfft) <= (2/3)*k_max*(2*math.pi/L))).unsqueeze(0)

    w_h, f_h = torch.fft.rfft2(w0_cr), torch.fft.rfft2(f_cr)
    steps, record_time = math.ceil(T/delta_t), math.floor(math.ceil(T/delta_t)/record_steps)
    sol_history = torch.zeros(w0_cr.shape[0], N_cr, N_cr, record_steps, device=w0_cr.device)

    c_record, c_closure = 0, 0
    current_closure_h = torch.zeros_like(w_h)

    for j in tqdm(range(steps), desc=f"Running Corrected {N_cr}x{N_cr} Sim"):
        if j % record_time == 0 and c_closure < closure_history_cr.shape[1]:
            Pi_phys = closure_history_cr[:, c_closure, ...]
            if torch.isnan(Pi_phys).any():
                # Handle NaN logic as before
                break
            current_closure_h = torch.fft.rfft2(Pi_phys)
            c_closure += 1

        psi_h = w_h / lap
        u_x_phys = torch.fft.irfft2(1j * k_y_rfft * psi_h, s=(N_cr, N_cr))
        u_y_phys = torch.fft.irfft2(-1j * k_x_rfft * psi_h, s=(N_cr, N_cr))
        w_x_phys = torch.fft.irfft2(1j * k_x_rfft * w_h, s=(N_cr, N_cr))
        w_y_phys = torch.fft.irfft2(1j * k_y_rfft * w_h, s=(N_cr, N_cr))
        N_h = dealias * torch.fft.rfft2(u_x_phys * w_x_phys + u_y_phys * w_y_phys)
        N_h_corrected = N_h + current_closure_h

        # [CHANGE] The time-stepping scheme now includes the linear friction term 'r'
        denominator = 1.0 + 0.5 * delta_t * (visc * lap + r)
        numerator = (1.0 - 0.5 * delta_t * (visc * lap + r)) * w_h - delta_t * N_h_corrected + delta_t * f_h
        w_h = numerator / denominator

        if torch.isnan(w_h).any():
            # Handle NaN logic as before
            break

        if j % record_time == 0 and c_record < record_steps:
            sol_history[..., c_record] = torch.fft.irfft2(w_h, s=(N_cr, N_cr))
            c_record += 1

    return sol_history.permute(0, 3, 1, 2)

def solve_les_with_model(
    w0_cr, f_cr, visc, r, T, delta_t, snapshot_indices, domain_size,
    score_model, sampler_fn, sample_every_n_steps=10
):
    """[REVISED] Saves snapshots at the exact indices specified by snapshot_indices."""
    L, N_cr, device, batch_size = domain_size, w0_cr.shape[-1], w0_cr.device, w0_cr.shape[0]

    # ... (Wavenumber and dealiasing setup is unchanged) ...
    k_max=N_cr//2; k_y_int=torch.cat((torch.arange(0,k_max),torch.arange(-k_max,0)),0).to(device); k_x_int=k_y_int.view(N_cr,1)
    k_y,k_x=k_y_int*(2*math.pi/L),k_x_int*(2*math.pi/L); k_x_rfft,k_y_rfft=k_x[...,:k_max+1],k_y.view(1,N_cr)[...,:k_max+1]
    lap=k_x_rfft**2+k_y_rfft**2; lap[0,0]=1.0
    dealias=((torch.abs(k_x_rfft)<=(2/3)*k_max*(2*math.pi/L))&(torch.abs(k_y_rfft)<=(2/3)*k_max*(2*math.pi/L))).unsqueeze(0)

    w_h, f_h = torch.fft.rfft2(w0_cr), torch.fft.rfft2(f_cr)
    steps = int(T/delta_t)

    num_snapshots = len(snapshot_indices)
    sol_history = torch.zeros(batch_size, N_cr, N_cr, num_snapshots, device=device)
    snapshot_indices_set = set(snapshot_indices)
    c_record = 0

    current_closure_h = torch.zeros_like(w_h)

    desc = f"Running LES with {score_model.__class__.__name__}"
    for j in tqdm(range(steps), desc=desc):
        if j % sample_every_n_steps == 0:
            with torch.no_grad():
                w_phys_condition = torch.fft.irfft2(w_h, s=(N_cr, N_cr))
                Pi_phys = sampler_fn(w_phys_condition, score_model)
                current_closure_h = torch.fft.rfft2(Pi_phys)

        psi_h=w_h/lap; u_p=torch.fft.irfft2(1j*k_y_rfft*psi_h,s=(N_cr,N_cr)); v_p=torch.fft.irfft2(-1j*k_x_rfft*psi_h,s=(N_cr,N_cr))
        wx_p=torch.fft.irfft2(1j*k_x_rfft*w_h,s=(N_cr,N_cr)); wy_p=torch.fft.irfft2(1j*k_y_rfft*w_h,s=(N_cr,N_cr))
        N_h=dealias*torch.fft.rfft2(u_p*wx_p+v_p*wy_p)
        N_h_corrected = N_h + current_closure_h

        denom=1.0+0.5*delta_t*(visc*lap+r); numer=(1.0-0.5*delta_t*(visc*lap+r))*w_h-delta_t*N_h_corrected+delta_t*f_h
        w_h=numer/denom

        if j in snapshot_indices_set:
            if c_record < num_snapshots:
                sol_history[..., c_record] = torch.fft.irfft2(w_h, s=(N_cr, N_cr))
                c_record += 1

    return sol_history.permute(0, 3, 1, 2)


def solve_les_with_deterministic_model(
    w0_cr, f_cr, visc, r, T, delta_t, snapshot_indices, domain_size,
    model, sample_every_n_steps=10
):
    """[REVISED] Saves snapshots at the exact indices specified by snapshot_indices."""
    L, N_cr, device, batch_size = domain_size, w0_cr.shape[-1], w0_cr.device, w0_cr.shape[0]

    # ... (Wavenumber and dealiasing setup is unchanged) ...
    k_max=N_cr//2; k_y_int=torch.cat((torch.arange(0,k_max),torch.arange(-k_max,0)),0).to(device); k_x_int=k_y_int.view(N_cr,1)
    k_y,k_x=k_y_int*(2*math.pi/L),k_x_int*(2*math.pi/L); k_x_rfft,k_y_rfft=k_x[...,:k_max+1],k_y.view(1,N_cr)[...,:k_max+1]
    lap=k_x_rfft**2+k_y_rfft**2; lap[0,0]=1.0
    dealias=((torch.abs(k_x_rfft)<=(2/3)*k_max*(2*math.pi/L))&(torch.abs(k_y_rfft)<=(2/3)*k_max*(2*math.pi/L))).unsqueeze(0)

    w_h, f_h = torch.fft.rfft2(w0_cr), torch.fft.rfft2(f_cr)
    steps = int(T/delta_t)

    num_snapshots = len(snapshot_indices)
    sol_history = torch.zeros(batch_size, N_cr, N_cr, num_snapshots, device=device)
    snapshot_indices_set = set(snapshot_indices)
    c_record = 0

    current_closure_h = torch.zeros_like(w_h)

    desc = f"Running LES with {model.__class__.__name__}"
    for j in tqdm(range(steps), desc=desc):
        if j % sample_every_n_steps == 0:
            with torch.no_grad():
                w_phys_condition = torch.fft.irfft2(w_h, s=(N_cr, N_cr))
                Pi_phys = model(w_phys_condition)
                current_closure_h = torch.fft.rfft2(Pi_phys)

        psi_h=w_h/lap; u_p=torch.fft.irfft2(1j*k_y_rfft*psi_h,s=(N_cr,N_cr)); v_p=torch.fft.irfft2(-1j*k_x_rfft*psi_h,s=(N_cr,N_cr))
        wx_p=torch.fft.irfft2(1j*k_x_rfft*w_h,s=(N_cr,N_cr)); wy_p=torch.fft.irfft2(1j*k_y_rfft*w_h,s=(N_cr,N_cr))
        N_h=dealias*torch.fft.rfft2(u_p*wx_p+v_p*wy_p)
        N_h_corrected = N_h + current_closure_h

        denom=1.0+0.5*delta_t*(visc*lap+r); numer=(1.0-0.5*delta_t*(visc*lap+r))*w_h-delta_t*N_h_corrected+delta_t*f_h
        w_h=numer/denom

        if j in snapshot_indices_set:
            if c_record < num_snapshots:
                sol_history[..., c_record] = torch.fft.irfft2(w_h, s=(N_cr, N_cr))
                c_record += 1

    return sol_history.permute(0, 3, 1, 2)



def define_test_filter(N, domain_size, device='cpu'):
    """
    Defines the Fourier-space 'test' filter for the dynamic procedure.
    It is conventionally defined as having a width of twice the grid filter, Delta_hat = 2*Delta.
    For a Gaussian, this corresponds to a delta_frac of 2.0 relative to its own grid.
    """
    return define_fourier_gaussian_kernel(N, domain_size, delta_frac=2.0, device=device)


def solve_les_dynamic_smagorinsky(w0_cr, f_cr, visc, r, T, delta_t, snapshot_indices, domain_size):
    """
    [REVISED] Runs a coarse-grid LES using the dynamic Smagorinsky model and
    saves snapshots at the exact indices specified by snapshot_indices.
    """
    L = domain_size
    N_cr = w0_cr.size()[-1]
    if len(w0_cr.shape) == 2: w0_cr = w0_cr.unsqueeze(0)
    if len(f_cr.shape) == 2: f_cr = f_cr.unsqueeze(0)

    # --- Wavenumbers and Operators for the Coarse Grid ---
    k_max = N_cr // 2
    k_y_int = torch.cat((torch.arange(0,k_max),torch.arange(-k_max,0)),0).to(w0_cr.device)
    k_x_int = k_y_int.view(N_cr, 1)
    k_y, k_x = k_y_int * (2*math.pi/L), k_x_int * (2*math.pi/L)
    k_x_rfft, k_y_rfft = k_x[..., :k_max+1], k_y.view(1,N_cr)[..., :k_max+1]
    lap = k_x_rfft**2+k_y_rfft**2
    lap[0,0] = 1.0
    dealias = ((torch.abs(k_x_rfft) <= (2/3)*k_max*(2*math.pi/L)) & (torch.abs(k_y_rfft) <= (2/3)*k_max*(2*math.pi/L))).unsqueeze(0)

    # --- Define Test Filter for the Dynamic Procedure ---
    test_filter_h = define_test_filter(N_cr, domain_size, device=w0_cr.device)
    delta_sq = (domain_size / N_cr)**2

    # --- Initialize Simulation State ---
    w_h, f_h = torch.fft.rfft2(w0_cr), torch.fft.rfft2(f_cr)
    steps = int(T/delta_t)

    # [CHANGE] Setup history tensors based on the number of indices to save
    num_snapshots = len(snapshot_indices)
    sol_history = torch.zeros(w0_cr.shape[0], N_cr, N_cr, num_snapshots, device=w0_cr.device)

    # [CHANGE] Use a set for fast 'in' check and a counter for saving
    snapshot_indices_set = set(snapshot_indices)
    c_record = 0

    for j in tqdm(range(steps), desc=f"Running Dynamic Smagorinsky ({N_cr}x{N_cr})"):
        # --- DYNAMIC PROCEDURE to find the closure term at this step ---
        psi_h = w_h / lap
        u_h = 1j * k_y_rfft * psi_h
        v_h = -1j * k_x_rfft * psi_h

        u_hat_h = u_h * test_filter_h
        v_hat_h = v_h * test_filter_h

        u = torch.fft.irfft2(u_h, s=(N_cr, N_cr))
        v = torch.fft.irfft2(v_h, s=(N_cr, N_cr))
        u_hat = torch.fft.irfft2(u_hat_h, s=(N_cr, N_cr))
        v_hat = torch.fft.irfft2(v_hat_h, s=(N_cr, N_cr))

        L11 = torch.fft.irfft2(torch.fft.rfft2(u * u) * test_filter_h, s=(N_cr,N_cr)) - u_hat * u_hat
        L12 = torch.fft.irfft2(torch.fft.rfft2(u * v) * test_filter_h, s=(N_cr,N_cr)) - u_hat * v_hat
        L22 = torch.fft.irfft2(torch.fft.rfft2(v * v) * test_filter_h, s=(N_cr,N_cr)) - v_hat * v_hat

        S11_h = 1j * k_x_rfft * u_h
        S12_h = 0.5 * (1j * k_y_rfft * u_h + 1j * k_x_rfft * v_h)
        S22_h = 1j * k_y_rfft * v_h
        S11 = torch.fft.irfft2(S11_h, s=(N_cr, N_cr))
        S12 = torch.fft.irfft2(S12_h, s=(N_cr, N_cr))
        S22 = torch.fft.irfft2(S22_h, s=(N_cr, N_cr))
        S_mag = torch.sqrt(2 * (S11**2 + 2 * S12**2 + S22**2))

        alpha_11 = delta_sq * S_mag * S11; alpha_12 = delta_sq * S_mag * S12; alpha_22 = delta_sq * S_mag * S22
        M11 = -2 * torch.fft.irfft2(torch.fft.rfft2(alpha_11) * test_filter_h, s=(N_cr,N_cr))
        M12 = -2 * torch.fft.irfft2(torch.fft.rfft2(alpha_12) * test_filter_h, s=(N_cr,N_cr))
        M22 = -2 * torch.fft.irfft2(torch.fft.rfft2(alpha_22) * test_filter_h, s=(N_cr,N_cr))

        L_M = L11*M11 + 2*L12*M12 + L22*M22
        M_M = M11*M11 + 2*M12*M12 + M22*M22
        Cs_sq = torch.mean(L_M, dim=(-1,-2), keepdim=True) / (torch.mean(M_M, dim=(-1,-2), keepdim=True) + 1e-12)
        Cs_sq = torch.clamp(Cs_sq, min=0.0)

        nu_t = Cs_sq * delta_sq * S_mag

        tau11 = -2 * nu_t * S11; tau12 = -2 * nu_t * S12; tau22 = -2 * nu_t * S22
        tau11_h, tau12_h, tau22_h = torch.fft.rfft2(tau11), torch.fft.rfft2(tau12), torch.fft.rfft2(tau22)
        div_tau_x_h = 1j*k_x_rfft*tau11_h + 1j*k_y_rfft*tau12_h
        div_tau_y_h = 1j*k_x_rfft*tau12_h + 1j*k_y_rfft*tau22_h
        current_closure_h = 1j*k_x_rfft*div_tau_y_h - 1j*k_y_rfft*div_tau_x_h

        # --- Standard Time Step ---
        w_x_phys = torch.fft.irfft2(1j * k_x_rfft * w_h, s=(N_cr, N_cr))
        w_y_phys = torch.fft.irfft2(1j * k_y_rfft * w_h, s=(N_cr, N_cr))
        N_h = dealias * torch.fft.rfft2(u * w_x_phys + v * w_y_phys)

        N_h_corrected = N_h + current_closure_h
        denominator = 1.0 + 0.5 * delta_t * (visc * lap + r)
        numerator = (1.0 - 0.5 * delta_t * (visc * lap + r)) * w_h - delta_t * N_h_corrected + delta_t * f_h
        w_h = numerator / denominator

        if torch.isnan(w_h).any():
            sol_history[..., c_record:] = float('nan')
            break

        # [CHANGE] Save the state if the current step is in our target indices
        if j in snapshot_indices_set:
            if c_record < num_snapshots:
                sol_history[..., c_record] = torch.fft.irfft2(w_h, s=(N_cr, N_cr))
                c_record += 1

    return sol_history.permute(0, 3, 1, 2)