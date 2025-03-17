import sys
sys.path.append('C:\\UWMadisonResearch\\Joint_LDM')
import h5py
import torch
from torch.optim import Adam
from functools import partial
from tqdm import trange
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utility import set_seed, fro_err, mse_err, get_sigmas_karras
from DiffusionModel import (marginal_prob_std, diffusion_coeff, loss_fn, FNO2d_Physics, FNO2d_Interp, FNO2d_Orig)

# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available.")
    device = torch.device('cuda')
else:
    print("CUDA is not available.")
    device = torch.device('cpu')

# Load the data

train_name = 'C:\\UWMadisonResearch\\Joint_LDM\\Data\\train_diffusion_nonlinear_v2.h5'
with h5py.File(train_name, 'r') as file:
    train_vorticity = torch.tensor(file['train_vorticity_64'][:], device=device)
    train_nonlinear = torch.tensor(file['train_nonlinear_64'][:], device=device)

test_name = 'C:\\UWMadisonResearch\\Joint_LDM\\Data\\test_diffusion_nonlinear_v2.h5'
with h5py.File(test_name, 'r') as file:
    test_vorticity = torch.tensor(file['test_vorticity_64'][:], device=device)
    test_nonlinear = torch.tensor(file['test_nonlinear_64'][:], device=device)

truncated_pc = 256

scalar = StandardScaler()
nonlinear = torch.cat([train_nonlinear, test_nonlinear], dim=0).cpu().numpy().reshape(-1, 64*64)
nonlinear_scaled = scalar.fit_transform(nonlinear)
train_nonlinear_scaled = nonlinear_scaled[:19000]
test_nonlinear_scaled = nonlinear_scaled[19000:]

pca = PCA(n_components=4096)
train_nonlinear_pca = pca.fit_transform(train_nonlinear_scaled)
test_nonlinear_pca = pca.transform(test_nonlinear_scaled)
cumsum = pca.explained_variance_ratio_.cumsum()

train_nonlinear_pca_truncated = train_nonlinear_pca[:, :truncated_pc]
test_nonlinear_pca_truncated = test_nonlinear_pca[:, :truncated_pc]

w_truncated = pca.components_[:truncated_pc]

train_nonlinear_recon = train_nonlinear_pca_truncated.dot(w_truncated) + pca.mean_
train_nonlinear_recon = scalar.inverse_transform(train_nonlinear_recon).reshape(-1, 64, 64)

test_nonlinear_recon = test_nonlinear_pca_truncated.dot(w_truncated) + pca.mean_
test_nonlinear_recon = scalar.inverse_transform(test_nonlinear_recon).reshape(-1, 64, 64)

mse_err_train = mse_err(train_nonlinear, torch.tensor(train_nonlinear_recon).to(device))
mse_err_test = mse_err(test_nonlinear, torch.tensor(test_nonlinear_recon).to(device))
fro_err_train = fro_err(train_nonlinear, torch.tensor(train_nonlinear_recon).to(device))
fro_err_test = fro_err(test_nonlinear, torch.tensor(test_nonlinear_recon).to(device))


vorticity_scalar = StandardScaler()
vorticity = torch.cat([train_vorticity, test_vorticity], dim=0).cpu().numpy().reshape(-1, 64*64)
vorticity_scaled = vorticity_scalar.fit_transform(vorticity)
train_vorticity_pca = pca.fit_transform(vorticity_scaled[:19000])
test_vorticity_pca = pca.transform(vorticity_scaled[19000:])

cumsum_vorticity = pca.explained_variance_ratio_.cumsum()

train_vorticity_pca_truncated = train_vorticity_pca[:, :truncated_pc]
test_vorticity_pca_truncated = test_vorticity_pca[:, :truncated_pc]

w_truncated_vorticity = pca.components_[:truncated_pc]

train_vorticity_recon = train_vorticity_pca_truncated.dot(w_truncated_vorticity) + pca.mean_
train_vorticity_recon = vorticity_scalar.inverse_transform(train_vorticity_recon).reshape(-1, 64, 64)

test_vorticity_recon = test_vorticity_pca_truncated.dot(w_truncated_vorticity) + pca.mean_
test_vorticity_recon = vorticity_scalar.inverse_transform(test_vorticity_recon).reshape(-1, 64, 64)

mse_err_train_vorticity = mse_err(train_vorticity, torch.tensor(train_vorticity_recon).to(device))
mse_err_test_vorticity = mse_err(test_vorticity, torch.tensor(test_vorticity_recon).to(device))
fro_err_train_vorticity = fro_err(train_vorticity, torch.tensor(train_vorticity_recon).to(device))
fro_err_test_vorticity = fro_err(test_vorticity, torch.tensor(test_vorticity_recon).to(device))


import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(test_vorticity_recon[0], ax=ax[0], cbar=True, cmap='rocket', square=True)
ax[0].set_title('Reconstructed')
ax[0].axis('off')
sns.heatmap(test_vorticity[0].cpu(), ax=ax[1], cbar=True, cmap='rocket', square=True)
ax[1].set_title('Original')
ax[1].axis('off')
plt.show()






train_nonlinear_pca_2d = torch.tensor(train_nonlinear_pca_truncated.reshape(-1, 16, 16), device=device)
train_vorticity_pca_2d = torch.tensor(train_vorticity_pca_truncated.reshape(-1, 16, 16), device=device)

test_nonlinear_pca_2d = torch.tensor(test_nonlinear_pca_truncated.reshape(-1, 16, 16), device=device)
test_vorticity_pca_2d = torch.tensor(test_vorticity_pca_truncated.reshape(-1, 16, 16), device=device)

import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(train_nonlinear_pca_2d[0].cpu(), ax=ax[0], cbar=True, cmap='rocket', square=True)
ax[0].set_title('Nonlinear PCA')
ax[0].axis('off')
sns.heatmap(train_vorticity_pca_2d[0].cpu(), ax=ax[1], cbar=True, cmap='rocket', square=True)
ax[1].set_title('Vorticity PCA')
ax[1].axis('off')
plt.show()


train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_nonlinear_pca_2d,
                                                                          train_vorticity_pca_2d),
                                                                            batch_size=100, shuffle=True)




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
sigma = 150
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

modes = 8
width = 20
epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

model = FNO2d_Orig(marginal_prob_std_fn, modes, modes, width).cuda()
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
    # rel_err = []
    for x, w in train_loader:
        x, w = x.cuda(), w.cuda()
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
torch.save(model.state_dict(), 'Convection_PCA_256.pth')


model.load_state_dict(torch.load('Convection_PCA_256.pth'))

sde_time_min = 1e-3
sde_time_max = 1
sample_steps = 100
sample_batch_size = 10

time_noises = get_sigmas_karras(sample_steps, sde_time_min, sde_time_max, device=device)


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

            score_err = torch.mean(torch.norm(grad - real_score, 2, dim=(1, 2))
                                   / torch.norm(real_score, 2, dim=(1, 2)))
            rel_err[i] = score_err
    return mean_x, rel_err

sample_batch_size = 10
sample_spatial_dim = 16

sampler = partial(sampler,
                  spatial_dim=sample_spatial_dim,
                marginal_prob_std = marginal_prob_std_fn,
                diffusion_coeff = diffusion_coeff_fn,
                batch_size = sample_batch_size,
                num_steps = sample_steps,
                time_noises = time_noises,
                device = device)

with torch.no_grad():
    test_sample, rel_err = sampler(test_nonlinear_pca_2d[:sample_batch_size],
                                    test_vorticity_pca_2d[:sample_batch_size], model)


test_sample_recon = test_sample.cpu().numpy().reshape(-1, 16*16).dot(w_truncated) + pca.mean_
test_sample_recon = scalar.inverse_transform(test_sample_recon).reshape(-1, 64, 64)

mse_err_test = mse_err(test_nonlinear_pca_2d[:sample_batch_size], test_sample)
fro_err_test = fro_err(test_nonlinear_pca_2d[:sample_batch_size], test_sample)

mse_err_test_sample = mse_err(test_nonlinear[:sample_batch_size], torch.tensor(test_sample_recon).to(device))
fro_err_test_sample = fro_err(test_nonlinear[:sample_batch_size], torch.tensor(test_sample_recon).to(device))


import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.heatmap(test_nonlinear_pca_2d[0].cpu(), ax=ax[0], cbar=True, cmap='rocket', square=True)
ax[0].set_title('Convection PC')
ax[0].axis('off')
sns.heatmap(test_vorticity_pca_2d[0].cpu(), ax=ax[1], cbar=True, cmap='rocket', square=True)
ax[1].set_title('Vorticity PC')
ax[1].axis('off')
plt.show()