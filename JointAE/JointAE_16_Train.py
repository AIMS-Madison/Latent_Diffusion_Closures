#!/usr/bin/env python
"""
train.py

"""

### 标准库
import os
import sys
import warnings

### 科学计算 & 深度学习库
import numpy as np
import torch
import h5py
from torch.optim import Adam
from functools import partial
from tqdm import trange

### 机器学习可视化库（仅在调试时使用）
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl

# 配置 matplotlib（如果在远程无GUI环境下运行，可注释掉显示相关代码）
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "Times New Roman"
mpl.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"

### 设置打印选项和忽略警告
np.set_printoptions(suppress=False, formatter={'float': '{:.2e}'.format})
torch.set_printoptions(sci_mode=True)
warnings.filterwarnings("ignore")

### 读取 OneDrive 路径（在 WSL 中建议手动设置环境变量）
onedrive_path = '/mnt/c/Users/dongx/OneDriveUWM'


### 检查 CUDA 是否可用
def get_device():
    if torch.cuda.is_available():
        print("✅ CUDA 可用，使用 GPU")
        return torch.device('cuda')
    else:
        print("❌ CUDA 不可用，使用 CPU")
        return torch.device('cpu')


device = get_device()


### 加载数据
def load_data():
    # 构造数据文件路径（确保路径中不要多余空格）
    train_name = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "Data",
                              "train_diffusion_nonlinear_sto_v5.h5")
    test_name = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "Data",
                             "test_diffusion_nonlinear_sto_v5.h5")

    print(f"Loading training data from {train_name}")
    with h5py.File(train_name, 'r') as file:
        train_nonlinear = torch.tensor(file['train_nonlinear_64'][:18000], device=device)
        train_vorticity = torch.tensor(file['train_vorticity_64'][:18000], device=device)

    print(f"Loading testing data from {test_name}")
    with h5py.File(test_name, 'r') as file:
        test_nonlinear = torch.tensor(file['test_nonlinear_64'][:], device=device)
        test_vorticity = torch.tensor(file['test_vorticity_64'][:], device=device)

    # 构造 DataLoader
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_nonlinear, train_vorticity),
        batch_size=50, shuffle=True
    )
    return train_loader


train_loader = load_data()

### 导入自定义模块（确保模块文件在当前工作目录或 PYTHONPATH 中）
from DiffusionModel import marginal_prob_std, diffusion_coeff, FNO2d_Orig, loss_fn
from utility import get_sigmas_karras, fro_err, mse_err, set_seed
from AE_Attention import VariationalAutoEncoder

### 设置噪声函数等
sigma = 30
marginal_prob_std_fn = partial(marginal_prob_std, sigma=sigma, device_=device)
diffusion_coeff_fn = partial(diffusion_coeff, sigma=sigma, device_=device)

### 超参数设置
modes = 4
width = 20
padding = 0
epochs = 500
learning_rate = 0.001
scheduler_step = 100
scheduler_gamma = 0.5

### 模型初始化
AEG_model = VariationalAutoEncoder().to(device)
AEW_model = VariationalAutoEncoder().to(device)
diffusion_model = FNO2d_Orig(marginal_prob_std_fn, modes, modes, width, padding, embed_dim=256, length=1).to(device)

AEG_path = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "PretrainAE", "AE_6416_vorticity_reg_sto_v5.pth")
# 如果需要加载预训练模型，可以取消下面的注释并调整路径
AEG_model.load_state_dict(torch.load(AEG_path, map_location=device))
# AEW_model.load_state_dict(torch.load('path_to_pretrain_AE_vorticity.pth'))

optimizer = Adam(list(diffusion_model.parameters()) + list(AEW_model.parameters()) + list(AEG_model.parameters()),
                 lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)


### 训练循环
def train():
    loss_history = []
    tqdm_epoch = trange(epochs, desc="Training")
    for epoch in tqdm_epoch:
        diffusion_model.train()
        AEW_model.train()
        AEG_model.train()
        total_loss = 0.0
        num_items = 0

        for x, w in train_loader:
            x, w = x.to(device), w.to(device)
            optimizer.zero_grad()

            # Autoencoder forward
            latent_x = AEG_model.encode(x)
            recon_x = AEG_model.decode(latent_x)
            fro_x = fro_err(x, recon_x)

            flattened_latent_x = latent_x.view(latent_x.shape[0], -1)
            latent_mean = flattened_latent_x.mean(dim=0)
            latent_var = flattened_latent_x.var(dim=0, unbiased=True)
            kl_divergence = 0.5 * (latent_var + latent_mean ** 2 - 1 - torch.log(latent_var + 1e-8))
            var_loss = kl_divergence.mean() * 0.1

            latent_w = AEW_model.encode(w)
            recon_w = AEW_model.decode(latent_w)
            fro_w = fro_err(w, recon_w)

            recon_loss_x = torch.nn.MSELoss()(recon_x, x) * 100
            recon_loss_w = torch.nn.MSELoss()(recon_w, w)

            score_loss, _, _ = loss_fn(diffusion_model, latent_x, latent_w, None, marginal_prob_std_fn, sparse=False)

            loss = score_loss + recon_loss_x + recon_loss_w + var_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]

        scheduler.step()
        avg_loss = total_loss / num_items
        loss_history.append(avg_loss)
        tqdm_epoch.set_description(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.5f}")

    # 保存模型到 OneDrive
    diffusion_model_save = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "JointAE",
                                        "Joint_diffusion_6416_sto_v5.pth")
    torch.save(diffusion_model.state_dict(), diffusion_model_save)

    AEG_model_save = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "JointAE",
                                  "Joint_AE_Nonlinear_6416_sto_v5.pth")
    torch.save(AEG_model.state_dict(), AEG_model_save)

    AEW_model_save = os.path.join(onedrive_path, "UWMadisonResearch", "Joint_LDM", "JointAE",
                                  "Joint_AE_Vorticity_6416_sto_v5.pth")
    torch.save(AEW_model.state_dict(), AEW_model_save)

    print("Training complete! Models saved.")


if __name__ == '__main__':
    # 为保证结果可重复（可选）
    set_seed(42)
    train()
