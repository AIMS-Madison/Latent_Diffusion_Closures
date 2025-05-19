import sys
sys.path.append('C:\\UWMadisonResearch\\Joint_LDM\\PretrainAE')
import h5py
import torch
from torch.optim import Adam
from functools import partial
from tqdm import trange
from utility import set_seed, fro_err, mse_err, get_sigmas_karras
from DiffusionModel import (marginal_prob_std, diffusion_coeff, loss_fn, FNO2d_Orig)

from AE_Attention import VariationalAutoEncoder

import numpy as np
np.set_printoptions(suppress=False, formatter={'float': '{:.2e}'.format})
torch.set_printoptions(sci_mode=True)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
plt.rc("text", usetex=True)
mpl.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

import warnings
warnings.filterwarnings("ignore")

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device('cuda')
else:
    print("CUDA is not available.")
    device = torch.device('cpu')

# Load the data

train_name = 'C:\\Users\\dongx\\OneDriveUWM\\UWMadisonResearch\\Joint_LDM\\Data\\train_diffusion_nonlinear_sto_v2.h5'
with h5py.File(train_name, 'r') as file:
    train_vorticity = torch.tensor(file['train_vorticity_64'][:], device=device)
    train_nonlinear = torch.tensor(file['train_nonlinear_64'][:], device=device)

test_name = 'C:\\Users\\dongx\\OneDriveUWM\\UWMadisonResearch\\Joint_LDM\\Data\\test_diffusion_nonlinear_sto_v2.h5'
with h5py.File(test_name, 'r') as file:
    test_vorticity = torch.tensor(file['test_vorticity_64'][:], device=device)
    test_nonlinear = torch.tensor(file['test_nonlinear_64'][:], device=device)


convection_AE = VariationalAutoEncoder().to(device)
convection_AE.load_state_dict(torch.load('C:\\Users\\dongx\\OneDriveUWM\\UWMadisonResearch\\Joint_LDM\\PretrainAE\\AE_6416_nonlinear_sto_v2_noKL.pth'))

vorticity_AE = VariationalAutoEncoder().to(device)
vorticity_AE.load_state_dict(torch.load('C:\\Users\\dongx\\OneDriveUWM\\UWMadisonResearch\\Joint_LDM\\PretrainAE\\AE_6416_vorticity_sto_v2_noKL.pth'))

# File to store the encoded outputs.
filename = r'C:\\Users\\dongx\\OneDriveUWM\\UWMadisonResearch\\Joint_LDM\Data\train_diffusion_nonlinear_encoded_sto_v2.h5'
batch_size = 1000
total_samples = 18000

# Determine the shape of one encoded sample.
with torch.no_grad():
    sample_vort = vorticity_AE.encode(train_vorticity[0:1])
    sample_nonlin = convection_AE.encode(train_nonlinear[0:1])

# Ensure the sample shape is (16, 16) (i.e. without channel dimension).
sample_shape_vort = sample_vort[0].shape  # expected: (16, 16)
sample_shape_nonlin = sample_nonlin[0].shape  # expected: (16, 16)
print("Encoded vorticity sample shape:", sample_shape_vort)
print("Encoded nonlinear sample shape:", sample_shape_nonlin)

# Create the HDF5 file and datasets.
with h5py.File(filename, 'w') as file:
    # Create chunked datasets with an initial size of 0 along the first axis.
    dset_vort = file.create_dataset(
        'train_vorticity_encoded',
        shape=(0,) + sample_shape_vort,
        maxshape=(total_samples,) + sample_shape_vort,
        chunks=(1,) + sample_shape_vort,
        dtype='double'
    )
    dset_nonlin = file.create_dataset(
        'train_nonlinear_encoded',
        shape=(0,) + sample_shape_nonlin,
        maxshape=(total_samples,) + sample_shape_nonlin,
        chunks=(1,) + sample_shape_nonlin,
        dtype='double'
    )

    # Process the data in batches.
    for i in range(0, total_samples, batch_size):
        # Process each batch on GPU (ensuring gradients are not tracked).
        with torch.no_grad():
            batch_vort = vorticity_AE.encode(
                train_vorticity[i:i + batch_size]
            )
            batch_nonlin = convection_AE.encode(
                train_nonlinear[i:i + batch_size]
            )

        # Convert to NumPy arrays (on CPU).
        batch_vort_np = batch_vort.cpu().numpy()
        batch_nonlin_np = batch_nonlin.cpu().numpy()

        # Get current size of the datasets along the first axis.
        cur_size = dset_vort.shape[0]
        new_size = cur_size + batch_vort_np.shape[0]

        # Resize datasets to accommodate the new batch.
        dset_vort.resize(new_size, axis=0)
        dset_nonlin.resize(new_size, axis=0)

        # Write the batch data.
        dset_vort[cur_size:new_size] = batch_vort_np
        dset_nonlin[cur_size:new_size] = batch_nonlin_np

        print(f"Appended samples {cur_size} to {new_size - 1}")

print("All batches appended successfully.")







# File to store the encoded outputs.
filename = r'C:\\Users\\dongx\\OneDriveUWM\\UWMadisonResearch\\Joint_LDM\Data\test_diffusion_nonlinear_encoded_sto_v2.h5'
batch_size = 1000
total_samples = 2000

# Determine the shape of one encoded sample.
with torch.no_grad():
    sample_vort = vorticity_AE.encode(test_vorticity[0:1])
    sample_nonlin = convection_AE.encode(test_nonlinear[0:1])

# Ensure the sample shape is (16, 16) (i.e. without channel dimension).
sample_shape_vort = sample_vort[0].shape  # expected: (16, 16)
sample_shape_nonlin = sample_nonlin[0].shape  # expected: (16, 16)
print("Encoded vorticity sample shape:", sample_shape_vort)
print("Encoded nonlinear sample shape:", sample_shape_nonlin)

# Create the HDF5 file and datasets.
with h5py.File(filename, 'w') as file:
    # Create chunked datasets with an initial size of 0 along the first axis.
    dset_vort = file.create_dataset(
        'test_vorticity_encoded',
        shape=(0,) + sample_shape_vort,
        maxshape=(total_samples,) + sample_shape_vort,
        chunks=(1,) + sample_shape_vort,
        dtype='double'
    )
    dset_nonlin = file.create_dataset(
        'test_nonlinear_encoded',
        shape=(0,) + sample_shape_nonlin,
        maxshape=(total_samples,) + sample_shape_nonlin,
        chunks=(1,) + sample_shape_nonlin,
        dtype='double'
    )

    # Process the data in batches.
    for i in range(0, total_samples, batch_size):
        # Process each batch on GPU (ensuring gradients are not tracked).
        with torch.no_grad():
            batch_vort = vorticity_AE.encode(
                test_vorticity[i:i + batch_size]
            )
            batch_nonlin = convection_AE.encode(
                test_nonlinear[i:i + batch_size]
            )

        # Convert to NumPy arrays (on CPU).
        batch_vort_np = batch_vort.cpu().numpy()
        batch_nonlin_np = batch_nonlin.cpu().numpy()

        # Get current size of the datasets along the first axis.
        cur_size = dset_vort.shape[0]
        new_size = cur_size + batch_vort_np.shape[0]

        # Resize datasets to accommodate the new batch.
        dset_vort.resize(new_size, axis=0)
        dset_nonlin.resize(new_size, axis=0)

        # Write the batch data.
        dset_vort[cur_size:new_size] = batch_vort_np
        dset_nonlin[cur_size:new_size] = batch_nonlin_np

        print(f"Appended samples {cur_size} to {new_size - 1}")

print("All batches appended successfully.")
#

#
# scalar = MinMaxScaler(feature_range=(-1, 1))
# train_nonlinear_encoded = scalar.fit_transform(train_nonlinear_encoded.cpu().reshape(-1, 16*16)).reshape(-1, 16, 16)
# test_nonlinear_encoded = scalar.transform(test_nonlinear_encoded.cpu().reshape(-1, 16*16)).reshape(-1, 16, 16)
#
# train_nonlinear_encoded = torch.tensor(train_nonlinear_encoded, device=device)
# test_nonlinear_encoded = torch.tensor(test_nonlinear_encoded, device=device)
#
# scalarVorticity = MinMaxScaler(feature_range=(-1, 1))
# train_vorticity_encoded = scalarVorticity.fit_transform(train_vorticity_encoded.cpu().reshape(-1, 16*16)).reshape(-1, 16, 16)
# test_vorticity_encoded = scalarVorticity.transform(test_vorticity_encoded.cpu().reshape(-1, 16*16)).reshape(-1, 16, 16)
#
# train_vorticity_encoded = torch.tensor(train_vorticity_encoded, device=device)
# test_vorticity_encoded = torch.tensor(test_vorticity_encoded, device=device)


encoded_train_name = 'C:\\UWMadisonResearch\\Joint_LDM\\Data\\train_diffusion_nonlinear_encoded_sto_v2.h5'
with h5py.File(encoded_train_name, 'r') as file:
    train_vorticity_encoded = torch.tensor(file['train_vorticity_encoded'][:], device=device)
    train_nonlinear_encoded = torch.tensor(file['train_nonlinear_encoded'][:], device=device)

encoded_test_name = 'C:\\UWMadisonResearch\\\Joint_LDM\\Data\\test_diffusion_nonlinear_encoded_sto_v2.h5'
with h5py.File(encoded_test_name, 'r') as file:
    test_vorticity_encoded = torch.tensor(file['test_vorticity_encoded'][:], device=device)
    test_nonlinear_encoded = torch.tensor(file['test_nonlinear_encoded'][:], device=device)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_nonlinear_encoded,
                                                                          train_vorticity_encoded),
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

################################
######## Model Training ########
################################
sigma = 30
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

modes = 4
width = 20
epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

model = FNO2d_Orig(marginal_prob_std_fn, modes, modes, width, embed_dim=256, length=1).cuda()
optimizer = Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

tqdm_epoch = trange(epochs)

loss_history = []
rel_err_history = []

set_seed(42)
for epoch in tqdm_epoch:
    model.train()
    avg_loss = 0.
    num_items = 0
    for x, w in train_loader:
        x, w = x.cuda(), w.cuda()
        x = x.float()
        w = w.float()
        optimizer.zero_grad()
        loss, _, _ = loss_fn(model, x, w, None, marginal_prob_std_fn, sparse=False)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * x.shape[0]
        num_items += x.shape[0]
        # relative_loss = torch.mean(torch.norm(score - real_score, 2, dim=(1, 2))
        #                            / torch.norm(real_score, 2, dim=(1, 2)))
        # rel_err.append(relative_loss.item())
    scheduler.step()
    avg_loss_epoch = avg_loss / num_items
    # relative_loss_epoch = np.mean(rel_err)
    loss_history.append(avg_loss_epoch)
    # rel_err_history.append(relative_loss_epoch)
    tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
torch.save(model.state_dict(), 'PretrainAE\\PretrainAE_Diffusion_reg_sto_v2.pth')


model.load_state_dict(torch.load('PretrainAE\\PretrainAE_Diffusion_reg_sto.pth'))

sde_time_min = 1e-3
sde_time_max = 0.1
sample_steps = 10
sample_batch_size = 100

time_noises = get_sigmas_karras(sample_steps, sde_time_min, sde_time_max, device=device)

time_noises = torch.linspace(sde_time_max, 0, sample_steps+1, device=device)

def sampler(target,
           vorticity_condition,
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
    rel_err = torch.zeros(num_steps)
    with (torch.no_grad()):
        for i in range(num_steps):
            batch_time_step = torch.ones(batch_size, device=device) * time_noises[i]
            real_score = -(x - target) / marginal_prob_std(batch_time_step)[:, None, None] ** 2
            step_size = time_noises[i] - time_noises[i + 1]
            g = diffusion_coeff(batch_time_step)
            grad = score_model(batch_time_step, x, vorticity_condition)

            mean_x = x + (g ** 2)[:, None, None] * grad * step_size
            x = mean_x + torch.sqrt(step_size) * g[:, None, None] * torch.randn_like(x)

            score_err = fro_err(real_score, grad)
            rel_err[i] = score_err
    return mean_x, rel_err

sample_spatial_dim = 16

sampler = partial(sampler,
                  spatial_dim=sample_spatial_dim,
                marginal_prob_std = marginal_prob_std_fn,
                diffusion_coeff = diffusion_coeff_fn,
                batch_size = sample_batch_size,
                num_steps = sample_steps,
                time_noises = time_noises,
                device = device)

test_nonlinear_encoded_ensemble = test_nonlinear_encoded[0:1, ...].repeat(sample_batch_size, 1, 1)
test_vorticity_encoded_ensemble = test_vorticity_encoded[0:1, ...].repeat(sample_batch_size, 1, 1)

# set_seed(42)
import time
start = time.time()
with torch.no_grad():
    train_sample, rel_err_train = sampler(train_nonlinear_encoded[:sample_batch_size],
                                    train_vorticity_encoded[:sample_batch_size].float(), model)
    test_sample, rel_err_test = sampler(test_nonlinear_encoded_ensemble[:sample_batch_size],
                                    test_vorticity_encoded_ensemble[:sample_batch_size].float(), model)
end = time.time()
print('Time elapsed: {}'.format(end - start))


fro_sample_train = fro_err(train_nonlinear_encoded[:sample_batch_size], train_sample)
mse_sample_train = mse_err(train_nonlinear_encoded[:sample_batch_size], train_sample)
fro_sample = fro_err(test_nonlinear_encoded_ensemble[:sample_batch_size], test_sample)
mse_sample = mse_err(test_nonlinear_encoded_ensemble[:sample_batch_size], test_sample)

ensemble_mean = torch.mean(test_sample, dim=0)
ensemble_std = torch.std(test_sample, dim=0)

ensemble_fro_err = fro_err(test_nonlinear_encoded_ensemble[0:1], ensemble_mean.unsqueeze(0))

start = time.time()
with torch.no_grad():
    decoded_train_sample = convection_AE.decode(train_sample.float())
    decoded_truth_sample = convection_AE.decode(test_nonlinear_encoded[:sample_batch_size].float())
    decoded_sample = convection_AE.decode(test_sample.float())

    decoded_vorticity = vorticity_AE.decode(test_vorticity_encoded[:sample_batch_size].float())
end = time.time()
print('Time elapsed: {}'.format(end - start))

fro_decoded_train = fro_err(train_nonlinear[:sample_batch_size], decoded_train_sample)
mse_decoded_train = mse_err(train_nonlinear[:sample_batch_size], decoded_train_sample)
fro_decoded = fro_err(test_nonlinear[:sample_batch_size], decoded_sample)
mse_decoded = mse_err(test_nonlinear[:sample_batch_size], decoded_sample)
fro_vorticity = fro_err(test_nonlinear[:sample_batch_size], decoded_truth_sample)



### Plot and save
set_seed(13)

data1 = test_nonlinear_encoded[:sample_batch_size, :, :].cpu()
data2 = test_sample.cpu()
data3 = test_nonlinear[:sample_batch_size, :, :].cpu()
data4 = decoded_sample.cpu()
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
indices = [torch.randint(0, data1.shape[0], (1,)).item() for _ in range(4)]

# Define color scale parameters
latent_max = 2.0
latent_min = -2.0
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
    'C:\\UWMadisonResearch\\Joint_LDM\\Plots\\ModelWithoutJoint.png',
    dpi=300,
    bbox_inches='tight'
)