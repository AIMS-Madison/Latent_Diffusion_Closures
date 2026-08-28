import sys
sys.path.append('C:\\UWMadisonResearch\\Generative_Closures_Geometry\\')

import torch
import numpy as np
import math
import h5py
from timeit import default_timer
from Data_Generation.generator_sns import navier_stokes_2d, navier_stokes_2d_closure
from Data_Generation.random_forcing import GaussianRF


filename = "./Data_Generation/"
device = torch.device('cuda')

# Viscosity parameter
nu = 1e-3

# Spatial Resolution
s = 64

# Temporal Resolution
T = 20
delta_t = 1e-3

# Number of solutions to generate
N = 1000

# Set up 2d GRF with covariance parameters
GRF = GaussianRF(2, s, alpha=2.5, tau=7, device=device)

# Forcing function: 0.1*(sin(2pi(x+y)) + cos(2pi(x+y)))
t = torch.linspace(0, 1, s + 1, device=device)
t = t[0:-1]

X, Y = torch.meshgrid(t, t)
f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))

# Stochastic forcing function: sigma*dW/dt
stochastic_forcing = {'alpha': 0.005, 'kappa': 10, 'sigma': 0.00005}

# Number of snapshots from solution
record_steps = 4

# Solve equations in batches (order of magnitude speed-up)
# Batch size
bsize = 1000

c = 0
t0 = default_timer()

# sol_col = torch.zeros(N, s, s, record_steps+1).to(device)
# sol_col_128 = torch.zeros(N, s//2, s//2, record_steps+1).to(device)
sol_col_64 = torch.zeros(N, s, s, record_steps+1).to(device)


diffusion_col = torch.zeros(N, s, s, record_steps+1).to(device)
nonlinear_col = torch.zeros(N, s, s, record_steps+1).to(device)
forcing_col = torch.zeros(N, s, s, record_steps+1).to(device)

for j in range(N // bsize):
    w0 = GRF.sample(bsize)
    sol, sol_t, diffusion_term, nonlinear_term, forcing_term = navier_stokes_2d([1, 1], w0, f, nu, T, delta_t, record_steps,
                                                                thres=0, stochastic_forcing = stochastic_forcing)

    c += bsize
    t1 = default_timer()
    print(j, c, t1 - t0)

    # sol_col[j * bsize:(j + 1) * bsize] = sol
    # sol_col_128[j * bsize:(j + 1) * bsize] = sol[:, ::2, ::2, :]
    sol_col_64[j * bsize:(j + 1) * bsize] = sol[:, :, :, :]
    nonlinear_col[j * bsize:(j + 1) * bsize] = nonlinear_term
    diffusion_col[j * bsize:(j + 1) * bsize] = diffusion_term
    forcing_col[j * bsize:(j + 1) * bsize] = forcing_term

train_vorticity_64 = sol_col_64[:90,..., :].permute(0,3,1,2).reshape(-1, s, s)
train_diffusion_64 = diffusion_col[:90,..., :].permute(0,3,1,2).reshape(-1, s, s)
train_nonlinear_64 = nonlinear_col[:90,..., :].permute(0,3,1,2).reshape(-1, s, s)

test_vorticity_64 = sol_col_64[90:100,..., :].permute(0,3,1,2).reshape(-1, s, s)
test_nonlinear_64 = nonlinear_col[90:100,..., :].permute(0,3,1,2).reshape(-1, s, s)
test_diffusion_64 = diffusion_col[90:100,..., :].permute(0,3,1,2).reshape(-1, s, s)

# test_vorticity_128 = sol_col_128[500:600,..., :100]
# test_nonlinear_128 = nonlinear_col_128[500:600,..., :100]
# test_vorticity_256 = sol_col[500:600,..., :100]
# test_nonlinear_256 = nonlinear_col[500:600,..., :100]

filename = 'train_diffusion_nonlinear.h5'
with h5py.File(filename, 'w') as file:
    file.create_dataset('t', data=sol_t.cpu().numpy())
    file.create_dataset('train_vorticity_64', data=train_vorticity_64.cpu().numpy())
    file.create_dataset('train_nonlinear_64', data=train_nonlinear_64.cpu().numpy())
    file.create_dataset('train_diffusion_64', data=train_diffusion_64.cpu().numpy())
    # save stochastic forcing configs as dictionary
    file.create_dataset('stochastic_forcing', data=np.array([stochastic_forcing['alpha'], stochastic_forcing['kappa'], stochastic_forcing['sigma']]))

filename = 'test_diffusion_nonlinear.h5'
with h5py.File(filename, 'w') as file:
    file.create_dataset('t', data=sol_t.cpu().numpy())
    file.create_dataset('test_vorticity_64', data=test_vorticity_64.cpu().numpy())
    file.create_dataset('test_nonlinear_64', data=test_nonlinear_64.cpu().numpy())
    file.create_dataset('test_diffusion_64', data=test_diffusion_64.cpu().numpy())
    
    # file.create_dataset('test_vorticity_128', data=test_vorticity_128.cpu().numpy())
    # file.create_dataset('test_nonlinear_128', data=test_nonlinear_128.cpu().numpy())
    # file.create_dataset('test_vorticity_256', data=test_vorticity_256.cpu().numpy())
    # file.create_dataset('test_nonlinear_256', data=test_nonlinear_256.cpu().numpy())