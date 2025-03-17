import torch
import torch.nn as nn
import h5py
import numpy as np
np.set_printoptions(suppress=False, formatter={'float': '{:.2e}'.format})
torch.set_printoptions(sci_mode=True)
from utility import set_seed, fro_err, mse_err
from AE_Attention import VariationalAutoEncoder, weights_init


# Load and prepare data
device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_name = 'C:\\UWMadisonResearch\\Joint_LDM\\Data\\train_diffusion_nonlinear_sto_v2.h5'
with h5py.File(train_name, 'r') as file:
    train_nonlinear = torch.tensor(file['train_nonlinear_64'][:], device=device)
    train_vorticity = torch.tensor(file['train_vorticity_64'][:], device=device)

test_name = 'C:\\UWMadisonResearch\\Joint_LDM\\Data\\test_diffusion_nonlinear_sto_v2.h5'
with h5py.File(test_name, 'r') as file:
    test_nonlinear = torch.tensor(file['test_nonlinear_64'][:], device=device)
    test_vorticity = torch.tensor(file['test_vorticity_64'][:], device=device)

train_loader = torch.utils.data.DataLoader(train_vorticity, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_vorticity, batch_size=10, shuffle=False)

# Usage example
model = VariationalAutoEncoder().to(device)
model.apply(weights_init)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
criterion = nn.MSELoss()

# Training loop with early stopping
num_epochs = 1000
patience = 1000
best_val_loss = float('inf')
counter = 0

recon_loss_history = torch.zeros(num_epochs)
var_loss_history = torch.zeros(num_epochs)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_fro = 0
    for batch in train_loader:
        inputs = batch.to(device)
        latent = model.encode(inputs)
        decoded = model.decode(latent)

        flattened_latent_x = latent.view(latent.shape[0], -1)
        latent_mean = flattened_latent_x.mean(dim=0)
        latent_var = flattened_latent_x.var(dim=0, unbiased=True)
        kl_divergence = 0.5 * (latent_var + latent_mean ** 2 - 1 - torch.log(
            latent_var + 1e-8))
        var_loss = kl_divergence.mean() * 1e-2

        recon_loss = criterion(decoded, inputs)  # Reconstruction loss (e.g., MSE)
        loss = recon_loss + var_loss # Total loss (reconstruction + regularization)

        recon_loss_history[epoch] = recon_loss

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_fro += fro_err(inputs, decoded)

    train_loss /= len(train_loader)
    train_fro /= len(train_loader)

    # Test
    model.eval()
    test_loss = 0
    test_fro = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch.to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, inputs).item()
            test_fro += fro_err(inputs, outputs)

    test_loss /= len(test_loader)
    test_fro /= len(test_loader)
    print(f'Epoch [{epoch + 1}/{num_epochs}] |',
          f'Last LR: {scheduler.get_last_lr()[0]:.6f} |',
          f'Train Loss: {train_loss:.6f} |',
          f'Train Fro Error: {train_fro:.6f} |',
          f'Recon Loss: {recon_loss:.6f} |',
            f'Var Loss: {var_loss:.6f} |',
          f'Test Loss: {test_loss:.6f}',
          f'Test Fro Error: {test_fro:.6f}')

    # Learning rate scheduler step
    scheduler.step(test_loss)
    torch.save(model.state_dict(), 'PretrainAE\\AE_6416_vorticity_reg_sto_v2.pth')
    #
    # # Early stopping based on validation loss
    # if test_loss < best_val_loss:
    #     best_val_loss = test_loss
    #     counter = 0
    #     # Save the best model's state
    #     torch.save(model.state_dict(), 'PretrainAE\\AE_6416_vorticity_reg_sto.pth')
    # else:
    #     counter += 1
    #     if counter >= patience:
    #         print("Early stopping")
    #         break
# Load best model
model.load_state_dict(torch.load('PretrainAE\\AE_6416_vorticity_reg_sto.pth'))

model.load_state_dict(torch.load('PretrainAE\\AE_6416_vorticity_reg_sto_v2.pth'))



# Evaluation on test set
model.eval()
test_loss = 0
test_re = 0

with torch.no_grad():
    test_batch = train_vorticity[0:100].to(device)
    test_output  = model(test_batch)
    latent = model.encode(test_batch)
    test_re = fro_err(test_batch, test_output)
    test_mse = mse_err(test_batch, test_output)





import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
plt.rc("text", usetex=True)
mpl.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

### Plot and save
set_seed(13)

data1 = test_nonlinear[:100, :, :].cpu()
data2 = latent[:100, :, :].cpu()
data3 = test_output[:100, :, :].cpu()

# Initialize the plot with 4 rows and 4 columns
fig, axs = plt.subplots(3, 2, figsize=(10, 12), constrained_layout=True)
fs = 26
plt.rcParams.update({'font.size': fs})

# Define tick positions and labels
def create_ticks_labels(size, step=20):
    ticks = np.arange(0, size, step * size / 64)
    tick_labels = [str(int(tick)) for tick in ticks]
    return ticks, tick_labels

ticks_1, tick_labels_1 = create_ticks_labels(data1.shape[1])
ticks_2, tick_labels_2 = create_ticks_labels(data2.shape[1])
ticks_3, tick_labels_3 = create_ticks_labels(data3.shape[1])

indices = [torch.randint(0, data1.shape[0], (1,)).item() for _ in range(2)]

# Plot heatmaps and contour plots
for i, idx in enumerate(indices):
    j = i % 2  # Column index

    # --- Row 1: Truth Heatmap ---
    truth = data1[idx, ...].cpu().numpy()
    max_val = truth.max()
    min_val = truth.min()
    cbar_ticks = np.linspace(min_val, max_val, 6)
    sns.heatmap(
        truth,
        ax=axs[0, j],
        cmap='rocket',
        vmax=max_val,
        vmin=min_val,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks, 'shrink': 1.0, 'aspect': 20},
        square=True
    )
    axs[0, j].set_title(r"\text{Truth }" + str(j + 1))
    axs[0, j].axis('off')  # Hide the axis

    # --- Row 2: Generated Heatmap ---
    encoded = data2[idx, ...].cpu().numpy()
    latent_max = encoded.max()
    latent_min = encoded.min()
    cbar_ticks_latent = np.linspace(latent_min, latent_max, 6)
    sns.heatmap(
        encoded,
        ax=axs[1, j],
        cmap='rocket',
        vmax=latent_max,
        vmin=latent_min,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks_latent, 'shrink': 1.0, 'aspect': 20},
        square=True
    )
    axs[1, j].set_title(r"\text{Encoded }" + str(j + 1))
    axs[1, j].axis('off')  # Hide the axis

    # --- Row 3: Error Heatmap ---
    reconstructed = data3[idx, ...].cpu().numpy()
    max_val = truth.max()
    min_val = truth.min()
    cbar_ticks = np.linspace(min_val, max_val, 6)
    sns.heatmap(
        reconstructed,
        ax=axs[2, j],
        cmap='rocket',
        vmax=max_val,
        vmin=min_val,
        cbar_kws={'format': '%.1f', 'ticks': cbar_ticks, 'shrink': 1.0, 'aspect': 20},
        square=True
    )
    axs[2, j].set_title(r"\text{Reconstructed }" + str(j + 1))
    axs[2, j].axis('off')  # Hide the axis

# Adjust layout and save the plot
# plt.subplots_adjust(right=0.85, hspace=0.3, wspace=0.5)
plt.show()



plt.savefig(
    'C:\\UWMadisonResearch\\Joint_LDM\\Plots\\NonlinearAE.png',
    dpi=300,
    bbox_inches='tight'
)