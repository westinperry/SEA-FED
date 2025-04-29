#!/usr/bin/env python3
import os, sys, torch, torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# project root
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, '..'))

import utils, data
from options.training_options import TrainOptions
from models.ae_3dconv import AutoEncoderCov3D, GatedAutoEncoderCov3D

opt = TrainOptions().parse(is_print=True)
device = torch.device("cuda" if opt.UseCUDA else "cpu")

# set seed + determinism
utils.seed(opt.Seed)
if opt.IsDeter:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# transforms & loader
frame_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])
train_idx = os.path.join(opt.DataRoot, opt.Dataset, 'Train_idx')
train_frm = os.path.join(opt.DataRoot, opt.Dataset, 'Train')
ds = data.VideoDataset(train_idx, train_frm, transform=frame_tf)
loader = DataLoader(ds, batch_size=opt.BatchSize, shuffle=True,
                    num_workers=opt.NumWorker, pin_memory=True)

# model
if opt.ModelName == 'AE':
    model = AutoEncoderCov3D(opt.ImgChnNum)
elif opt.ModelName == 'Gated_AE':
    model = GatedAutoEncoderCov3D(opt.ImgChnNum)
else:
    raise ValueError(f"Unknown model type: {opt.ModelName}")
model.to(device)

# resume if available
if opt.IsResume and opt.ResumePath:
    print(f"Loading model from {opt.ResumePath}")
    model.load_state_dict(torch.load(opt.ResumePath, map_location=device))
else:
    model.apply(utils.weights_init)

loss_fn = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=opt.LR)

# training loop
loss_per_epoch = []
for epoch in range(opt.EpochNum):
    model.train()
    total = 0
    for _, frames in loader:
        frames = frames.to(device)
        recon = model(frames)
        loss = loss_fn(recon, frames)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Clamp alpha gate to [0,1] if it exists
        if hasattr(model, 'alpha'):
            with torch.no_grad():
                model.alpha.clamp_(0.0, 1.0)

        total += loss.item()
    epoch_loss = total / len(loader)
    loss_per_epoch.append(epoch_loss)
    print(f"✅ Epoch {epoch+1}/{opt.EpochNum} — Avg Loss: {epoch_loss:.6f}")
    for n, p in model.named_parameters():
        if 'alpha' in n:
            print(f"{n}: {p.item():.4f}")

        

# save model
os.makedirs(opt.ModelRoot, exist_ok=True)
out = os.path.join(opt.ModelRoot, opt.OutputFile)
torch.save(model.state_dict(), out)
print(f"Model saved to {out}")

# optional plot
if opt.PlotGraph:
    plt.plot(range(1, opt.EpochNum + 1), loss_per_epoch, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    p = os.path.join(opt.ModelRoot, 'loss_curve.png')
    plt.savefig(p)
    print(f"Loss curve → {p}")
