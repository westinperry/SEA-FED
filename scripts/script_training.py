import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Add project root to PYTHONPATH
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import utils
import data
from options.training_options import TrainOptions
from models.ae_3dconv import AutoEncoderCov3D, GatedAutoEncoderCov3D

# Parse training options
opt_parser = TrainOptions()
opt = opt_parser.parse(is_print=True)

use_cuda = opt.UseCUDA
device = torch.device("cuda" if use_cuda else "cpu")

# Set seed
utils.seed(opt.Seed)
if opt.IsDeter:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Setup model configuration
model_setting = utils.get_model_setting(opt)
print(f'Setting: {model_setting}, bs={opt.BatchSize}, lr={opt.LR}')

# Setup dataset paths
dataset_root = opt.DataRoot
dataset_name = opt.Dataset
train_frame_dir = os.path.join(dataset_root, dataset_name, 'Train')
train_index_dir = os.path.join(dataset_root, dataset_name, 'Train_idx')

# Model saving path
saving_model_path = os.path.join(opt.ModelRoot, opt.OutputFile)
utils.mkdir(opt.ModelRoot)

# TensorBoard logger
tb_logger = utils.Logger(opt.ModelRoot) if opt.IsTbLog else None

# Transforms
frame_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
unorm_trans = utils.UnNormalize(mean=[0.5], std=[0.5])

# DataLoader
video_dataset = data.VideoDataset(train_index_dir, train_frame_dir, transform=frame_trans)
train_loader = DataLoader(video_dataset, batch_size=opt.BatchSize, shuffle=True, num_workers=opt.NumWorker, pin_memory=True)

# Model
if opt.ModelName == 'AE':
    model = AutoEncoderCov3D(opt.ImgChnNum)
elif opt.ModelName == 'Gated_AE':
    model = GatedAutoEncoderCov3D(opt.ImgChnNum)
else:
    raise ValueError("Unknown ModelName")
if opt.IsResume and opt.ResumePath:
    print(f"Loading model weights from: {opt.ResumePath}")
    model.load_state_dict(torch.load(opt.ResumePath, map_location=device))
else:
    model.apply(utils.weights_init)
model.to(device)

# Loss and Optimizer
loss_func = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.LR)

# Training Loop
print(f"Training {opt.EpochNum} epochs...")
loss_per_epoch = []
for epoch in range(opt.EpochNum):
    total_loss = 0
    for batch_idx, (_, frames) in enumerate(train_loader):
        frames = frames.to(device, non_blocking=True)
        recon = model(frames)
        loss = loss_func(recon, frames)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if tb_logger and batch_idx % opt.TextLogInterval == 0:
            tb_logger.scalar_summary('loss', loss.item(), epoch * len(train_loader) + batch_idx)

        if batch_idx % opt.TextLogInterval == 0:
            print(f"Epoch {epoch+1}/{opt.EpochNum}, Batch {batch_idx}: Loss = {loss.item():.6f}")

    epoch_loss = total_loss / len(train_loader)
    loss_per_epoch.append(epoch_loss)
    print(f"✅ Epoch {epoch+1} completed. Avg Loss: {epoch_loss:.6f}")

# Save final model
torch.save(model.state_dict(), saving_model_path)
print(f"Model saved to {saving_model_path}")

# Plot loss
if opt.PlotGraph:
    plt.plot(range(1, opt.EpochNum + 1), loss_per_epoch, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs Epoch')
    plt.grid(True)
    plot_path = os.path.join(opt.ModelRoot, 'loss_curve.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Loss curve saved to {plot_path}")
