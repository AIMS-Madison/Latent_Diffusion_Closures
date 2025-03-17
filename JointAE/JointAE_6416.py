import sys
sys.path.append('C:\\UWMadisonResearch\\Joint_LDM')
import h5py
import torch
from torch.optim import Adam
from functools import partial
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
plt.rc("text", usetex=True)
mpl.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

import numpy as np
np.set_printoptions(suppress=False, formatter={'float': '{:.2e}'.format})
torch.set_printoptions(sci_mode=True)

import warnings
warnings.filterwarnings("ignore")

from DiffusionModel import (marginal_prob_std, diffusion_coeff, FNO2d_Orig, loss_fn)
from utility import get_sigmas_karras, fro_err, mse_err, set_seed
from AE_Attention import VariationalAutoEncoder

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device('cuda')
else:
    print("CUDA is not available.")
    device = torch.device('cpu')

train_name = 'C:\\UWMadisonResearch\\Joint_LDM\\Data\\train_diffusion_nonlinear_sto_v2.h5'
with h5py.File(train_name, 'r') as file:
    train_nonlinear = torch.tensor(file['train_nonlinear_64'][:18000], device=device)
    train_vorticity = torch.tensor(file['train_vorticity_64'][:18000], device=device)

test_name = 'C:\\UWMadisonResearch\\Joint_LDM\\Data\\test_diffusion_nonlinear_sto_v2.h5'
with h5py.File(test_name, 'r') as file:
    test_nonlinear = torch.tensor(file['test_nonlinear_64'][:], device=device)
    test_vorticity = torch.tensor(file['test_vorticity_64'][:], device=device)



train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_nonlinear,
                                                                            train_vorticity),
                                                                            batch_size=50, shuffle=True)




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


sigma = 30
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

modes = 4
width = 20
padding = 0
epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

AEG_model = VariationalAutoEncoder().to(device)
AEW_model = VariationalAutoEncoder().to(device)
diffusion_model = FNO2d_Orig(marginal_prob_std_fn, modes, modes, width, padding, embed_dim = 256, length = 1).to(device)


AEG_model.load_state_dict(torch.load('C:\\UWMadisonResearch\\Joint_LDM\\PretrainAE\\AE_6416_nonlinear_reg_sto_v2.pth'))
AEW_model.load_state_dict(torch.load('C:\\UWMadisonResearch\\Joint_LDM\\PretrainAE\\AE_6416_vorticity_reg_sto_v2.pth'))
# AEG_model.load_state_dict(torch.load('PretrainAE\\AE_6416_nonlinear_reg_v3.pth'))
# AEW_model.load_state_dict(torch.load('PretrainAE\\AE_6416_vorticity_reg_v3.pth'))

optimizer = Adam(list(diffusion_model.parameters())+list(AEW_model.parameters())+list(AEG_model.parameters()),
                 lr=learning_rate)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

tqdm_epoch = trange(epochs)
loss_history = []
recon_err_x_history = []
recon_err_w_history = []
score_err_history = []
var_err_history = []
fro_x_history = []
fro_w_history = []
eps=1e-5

AE_criterion = torch.nn.MSELoss()

for epoch in tqdm_epoch:
    diffusion_model.train()
    AEW_model.train()
    AEG_model.train()
    avg_loss = 0.
    num_items = 0
    avg_recon_loss_x = 0.
    avg_recon_loss_w = 0.
    avg_score_loss = 0.
    avg_var_loss = 0.
    avg_fro_x = 0.
    avg_fro_w = 0.

    for x, w in train_loader:
        x = x.cuda()
        w = w.cuda()
        optimizer.zero_grad()

        latent_x = AEG_model.encode(x)
        recon_x = AEG_model.decode(latent_x)
        with torch.no_grad():
            fro_x = fro_err(x, recon_x)

        flattened_latent_x = latent_x.view(latent_x.shape[0], -1)
        latent_mean = flattened_latent_x.mean(dim=0)
        latent_var = flattened_latent_x.var(dim=0, unbiased=True)
        kl_divergence = 0.5 * (latent_var + latent_mean ** 2 - 1 - torch.log(
            latent_var + 1e-8))
        var_loss = kl_divergence.mean() * 0.1

        latent_w = AEW_model.encode(w)
        recon_w = AEW_model.decode(latent_w)

        with torch.no_grad():
            fro_w = fro_err(w, recon_w)

        reconloss_x = AE_criterion(recon_x, x) * 100
        reconloss_w = AE_criterion(recon_w, w)

        score_loss, _, _ =loss_fn(diffusion_model, latent_x, latent_w,
                                  None, marginal_prob_std_fn, sparse=False)

        loss = score_loss + reconloss_x + reconloss_w + var_loss

        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        avg_recon_loss_x += reconloss_x.item() * x.shape[0]
        avg_recon_loss_w += reconloss_w.item() * x.shape[0]
        avg_score_loss += score_loss.item() * x.shape[0]
        avg_var_loss += var_loss * x.shape[0]
        avg_fro_x += fro_x.item() * x.shape[0]
        avg_fro_w += fro_w.item() * x.shape[0]

        num_items += x.shape[0]

    scheduler.step()

    avg_loss_epoch = avg_loss / num_items
    avg_recon_loss_x_epoch = avg_recon_loss_x / num_items
    avg_recon_loss_w_epoch = avg_recon_loss_w / num_items
    avg_score_loss_epoch = avg_score_loss / num_items
    avg_var_loss_epoch = avg_var_loss / num_items
    avg_fro_x_epoch = avg_fro_x / num_items
    avg_fro_w_epoch = avg_fro_w / num_items

    loss_history.append(avg_loss_epoch)
    recon_err_x_history.append(avg_recon_loss_x_epoch)
    recon_err_w_history.append(avg_recon_loss_w_epoch)
    score_err_history.append(avg_score_loss_epoch)
    var_err_history.append(avg_var_loss_epoch)
    fro_x_history.append(avg_fro_x_epoch)
    fro_w_history.append(avg_fro_w_epoch)

    tqdm_epoch.set_description(
        f"Average Loss: {avg_loss / num_items:.5f} | "
        f"X Recon Loss: {avg_recon_loss_x / num_items:.5f} | "
        f"X Fro Loss: {avg_fro_x / num_items:.5f} | "
        f"W Recon Loss: {avg_recon_loss_w / num_items:.5f} | "
        f"W Fro Loss: {avg_fro_w / num_items:.5f} | "
        f"Score Loss: {avg_score_loss / num_items:.5f} | "
        f"Var Loss: {avg_var_loss / num_items:.5f} | "
    )

torch.save(diffusion_model.state_dict(), 'JointAE\\Joint_diffusion_6416_sto_v2.pth')
torch.save(AEW_model.state_dict(), 'JointAE\\Joint_AE_Vorticity_6416_sto_v2.pth')
torch.save(AEG_model.state_dict(), 'JointAE\\Joint_AE_Nonlinear_6416_sto_v2.pth')


var_err_history = torch.tensor(var_err_history).cpu().numpy()
i = 10
fig = plt.figure(figsize=(10, 6))
plt.plot(loss_history[i:])
plt.plot(recon_err_x_history[i:])
plt.plot(recon_err_w_history[i:])
plt.plot(score_err_history[i:])
plt.plot(var_err_history[i:])
plt.legend(['Total Loss', 'Recon Loss X', 'Recon Loss W', 'Score Loss', 'Var Loss'])
plt.show()


AEG_model.load_state_dict(torch.load('JointAE\\Joint_AE_Nonlinear_6416_sto.pth'))
AEW_model.load_state_dict(torch.load('JointAE\\Joint_AE_Vorticity_6416_sto.pth'))
diffusion_model.load_state_dict(torch.load('JointAE\\Joint_diffusion_6416_sto.pth'))


AEG_model.eval()
AEW_model.eval()
diffusion_model.eval()










sde_time_min = 1e-3
sde_time_max = 0.1
steps = 10

time_noises = get_sigmas_karras(steps, sde_time_min, sde_time_max, device=device)

time_noises = torch.linspace(sde_time_max, 0, steps+1, device=device)


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

sample_batch_size = 1000
sample_spatial_dim = 16

sampler = partial(sampler,
                  spatial_dim=sample_spatial_dim,
                marginal_prob_std = marginal_prob_std_fn,
                diffusion_coeff = diffusion_coeff_fn,
                batch_size = sample_batch_size,
                num_steps = steps,
                time_noises = time_noises,
                device = device)

import time

start = time.time()
with torch.no_grad():
    test_vorticity_latent = AEW_model.encode(test_vorticity[:sample_batch_size])
    sample_test = sampler(test_vorticity_latent, diffusion_model)
    sample_test_pixel = AEG_model.decode(sample_test)
end = time.time()
print('Time elapsed: {}'.format(end - start))

fro_err_sample = fro_err(test_nonlinear[:sample_batch_size], sample_test_pixel)



set_seed(42)

import time
start_time = time.time()
with torch.no_grad():
    test_vorticity_latent = AEW_model.encode(test_vorticity[:sample_batch_size])
end_time = time.time()
print("Time taken to encode vorticity: ", end_time - start_time)

start_time = time.time()
with torch.no_grad():
    test_nonlinear_latent = AEG_model.encode(test_nonlinear[:sample_batch_size])
end_time = time.time()
print("Time taken to encode nonlinear: ", end_time - start_time)


torch.cuda.synchronize()
start_time = time.time()
with torch.no_grad():
    sample_test = sampler(test_vorticity_latent, diffusion_model)
torch.cuda.synchronize()
end_time = time.time()
print("Time taken to sample: ", end_time - start_time)

start_time = time.time()
with torch.no_grad():
    sample_test_pixel = AEG_model.decode(sample_test)
end_time = time.time()
print("Time taken to decode sample: ", end_time - start_time)

with torch.no_grad():
    truth_test_pixel = AEG_model.decode(test_nonlinear_latent)
    test_vorticity_pixel = AEW_model.decode(test_vorticity_latent)

fro_err_sample = fro_err(test_nonlinear_latent, sample_test)
mse_err_sample = mse_err(test_nonlinear_latent, sample_test)

fro_err_phy_test = fro_err(test_nonlinear[:sample_batch_size], sample_test_pixel)
mse_err_phy_test = mse_err(test_nonlinear[:sample_batch_size], sample_test_pixel)

fro_err_AE_vorticity = fro_err(test_vorticity[:sample_batch_size], test_vorticity_pixel)
mse_err_AE_vorticity = mse_err(test_vorticity[:sample_batch_size], test_vorticity_pixel)
fro_err_AE_nonlinear = fro_err(test_nonlinear[:sample_batch_size], truth_test_pixel)
mse_err_AE_nonlinear = mse_err(test_nonlinear[:sample_batch_size], truth_test_pixel)


### Plot and save
set_seed(13)

data1 = test_nonlinear_latent[:sample_batch_size, :, :].cpu()
data2 = sample_test.cpu()
data3 = test_nonlinear[:sample_batch_size, :, :].cpu()
data4 = sample_test_pixel.cpu()
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
latent_max = 0.5
latent_min = -0.5
max_val = 0.7
min_val = -0.8
err_max = 0.1
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



