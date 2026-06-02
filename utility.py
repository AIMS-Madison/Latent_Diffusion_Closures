import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import random
import os


################################
##### Data Preprosessing #######
################################
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


################################
########### Sampling ###########
################################
def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])
def get_sigmas_karras(n, time_min, time_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n)
    min_inv_rho = time_min ** (1 / rho)
    max_inv_rho = time_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


################################
####### Energy Spectrum ########
################################
def moving_average(data, window_size):
    """ Simple moving average """
    return np.convolve(data, np.ones(window_size), 'valid') / window_size

def energy_spectrum(phi, lx=1, ly=1, smooth=True):
    # Assuming phi is of shape (time_steps, nx, ny)
    nx, ny = phi.shape[1], phi.shape[2]
    nt = nx * ny

    phi_h = np.fft.fftn(phi, axes=(1, 2)) / nt  # Fourier transform along spatial dimensions

    energy_h = 0.5 * (phi_h * np.conj(phi_h)).real  # Spectral energy density

    k0x = 2.0 * np.pi / lx
    k0y = 2.0 * np.pi / ly
    knorm = (k0x + k0y) / 3.0

    kxmax = nx // 2
    kymax = ny // 2

    wave_numbers = knorm * np.arange(0, nx)

    energy_spectrum = np.zeros(len(wave_numbers))

    for kx in range(nx):
        rkx = kx if kx <= kxmax else kx - nx
        for ky in range(ny):
            rky = ky if ky <= kymax else ky - ny
            rk = np.sqrt(rkx ** 2 + rky ** 2)
            k = int(np.round(rk))
            if k < len(wave_numbers):
                energy_spectrum[k] += np.sum(energy_h[:, kx, ky])

    energy_spectrum /= knorm

    if smooth:
        smoothed_spectrum = moving_average(energy_spectrum, 5)  # Smooth the spectrum
        smoothed_spectrum = np.append(smoothed_spectrum, np.zeros(4))  # Append zeros to match original length after convolution
        smoothed_spectrum[:4] = np.sum(energy_h[:, :4, :4].real, axis=(0, 1, 2)) / (knorm * phi.shape[0])  # First 4 values corrected
        energy_spectrum = smoothed_spectrum

    return {
        'k': wave_numbers,
        'E': energy_spectrum
    }

def energy_spectrum_tke_from_vorticity(phi, lx=1.0, ly=1.0, smooth=True, window_size=5):
    """
    Compute the isotropic 2D kinetic-energy spectrum from vorticity snapshots.

    Parameters
    ----------
    phi : np.ndarray
        Vorticity field of shape (time_steps, nx, ny).
    lx, ly : float
        Domain lengths in x and y.
    smooth : bool
        Whether to smooth the shell spectrum with a moving average.
    window_size : int
        Window size for moving average smoothing.

    Returns
    -------
    dict with keys:
        'k' : 1D array of shell-center wavenumbers
        'E' : 1D array of time-averaged isotropic TKE spectrum
    """
    # phi shape: (nt, nx, ny)
    nt, nx, ny = phi.shape
    nxy = nx * ny

    # Fourier transform of vorticity
    omega_hat = np.fft.fftn(phi, axes=(1, 2)) / nxy

    # Physical wavenumbers
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K = np.sqrt(K2)

    # Avoid division by zero at k = 0
    K2_nozero = K2.copy()
    K2_nozero[0, 0] = 1.0

    # Spectral kinetic energy density:
    # E(kx,ky) = 0.5 * (|u_hat|^2 + |v_hat|^2)
    #          = 0.5 * |omega_hat|^2 / |k|^2
    energy_h = 0.5 * (np.abs(omega_hat)**2) / K2_nozero[None, :, :]
    energy_h[:, 0, 0] = 0.0

    # Shell spacing: use the smallest fundamental increment
    dkx = 2.0 * np.pi / lx
    dky = 2.0 * np.pi / ly
    dk = min(dkx, dky)

    # Integer shell index
    shell_idx = np.rint(K / dk).astype(int)
    kmax = shell_idx.max()

    spectrum = np.zeros(kmax + 1)

    # Time-average shell spectrum
    for s in range(kmax + 1):
        mask = (shell_idx == s)
        spectrum[s] = np.mean(np.sum(energy_h[:, mask], axis=1))

    wave_numbers = dk * np.arange(kmax + 1)

    if smooth and len(spectrum) >= window_size:
        smoothed = moving_average(spectrum, window_size)
        pad = len(spectrum) - len(smoothed)
        smoothed = np.concatenate([smoothed, np.zeros(pad)])

        # keep the first few low-k entries unsmoothed
        keep = min(window_size - 1, len(spectrum))
        smoothed[:keep] = spectrum[:keep]
        spectrum = smoothed

    return {
        'k': wave_numbers,
        'E': spectrum
    }

def mse_err(data1, data2):
    return torch.mean((data1 - data2) ** 2)

def fro_err(data1, data2):
    error_fro = torch.linalg.matrix_norm(data1 - data2, 'fro', dim=(1, 2))
    return torch.mean(error_fro / torch.linalg.matrix_norm(data1, 'fro', dim=(1, 2)))

