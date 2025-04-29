#!/usr/bin/env python3
import os, sys, torch, torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# project root
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, '..'))

import utils, data
from options.training_options import TrainOptions
from models.ae_3dconv import AutoEncoderCov3D, GatedAutoEncoderCov3D

# --- Save Visualizations Helper ---
def save_feature_visualizations(features, save_dir, batch_idx, round_num, client_id):
    os.makedirs(save_dir, exist_ok=True)
    for name, feat in features.items():
        # Handle shape (B, C, D, H, W) or (B, C, H, W)
        if feat.dim() == 5:
            feat = feat.squeeze(2)  # remove D dimension if exists
        elif feat.dim() == 4:
            pass  # already (B, C, H, W)
        else:
            raise ValueError(f"Unexpected feature shape: {feat.shape}")

        sample = feat[0]  # (C, H, W)

        num_channels = min(8, sample.shape[0])
        selected_channels = torch.linspace(0, sample.shape[0]-1, steps=num_channels).long()

        fig, axs = plt.subplots(1, num_channels, figsize=(15, 3))
        for i, ch in enumerate(selected_channels):
            img = sample[ch].cpu()
            if img.dim() == 3:
                img = img[0]  # Take first slice if depth remains
            axs[i].imshow(img, cmap='gray')
            axs[i].axis('off')
            axs[i].set_title(f'Ch {ch.item()}')
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'client{client_id}_{name}_round{round_num}_batch{batch_idx}.png')
        plt.savefig(save_path)
        plt.close()
        print(f"🖼️ Saved: {save_path}")



# --- Start Training ---

opt = TrainOptions().parse()
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

# save folder for latent and features
feature_save_dir = os.path.join(opt.ModelRoot, "features", f"round{opt.Round}")
os.makedirs(feature_save_dir, exist_ok=True)

# training loop
loss_per_epoch = []
for epoch in range(opt.EpochNum):
    model.train()
    total = 0
    for batch_idx, (_, frames) in enumerate(loader):
        frames = frames.to(device)

        recon = model(frames)
        loss = loss_fn(recon, frames)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

        # (Optional) Print logs
        if batch_idx % opt.TextLogInterval == 0:
            print(f"[Round {opt.Round}] [Epoch {epoch+1}] Batch {batch_idx}: Loss = {loss.item():.6f}")

        # Save latent and SE/Adapter outputs every 100 batches
        if batch_idx % 100 == 0:
            with torch.no_grad():
                if isinstance(model, GatedAutoEncoderCov3D):
                    # Gated AE: extract full pathway
                    e1 = model.enc1(frames)
                    se1_out = model.se1(e1)
                    adapter1_out = model.adapter1(e1)
                    e1 = se1_out + adapter1_out

                    e2 = model.enc2(e1)
                    se2_out = model.se2(e2)
                    adapter2_out = model.adapter2(e2)
                    e2 = se2_out + adapter2_out

                    e3 = model.enc3(e2)
                    se3_out = model.se3(e3)
                    adapter3_out = model.adapter3(e3)
                    e3 = se3_out + adapter3_out

                    e4 = model.enc4(e3)
                    se4_out = model.se4(e4)
                    adapter4_out = model.adapter4(e4)
                    e4 = se4_out + adapter4_out

                    latent = e4
                    features = {
                        'latent': latent,
                        'enc1_se': se1_out,
                        'enc1_adapter': adapter1_out,
                        'enc2_se': se2_out,
                        'enc2_adapter': adapter2_out,
                        'enc3_se': se3_out,
                        'enc3_adapter': adapter3_out,
                        'enc4_se': se4_out,
                        'enc4_adapter': adapter4_out,
                    }

                elif isinstance(model, AutoEncoderCov3D):
                    # AutoEncoder: just extract encoder output
                    with torch.no_grad():
                        latent = model.encoder(frames)

                    features = {
                        'latent': latent
                    }


                save_feature_visualizations(features, feature_save_dir, batch_idx, opt.Round, opt.ClientID)


    epoch_loss = total / len(loader)
    loss_per_epoch.append(epoch_loss)
    print(f"✅ [Round {opt.Round}] Epoch {epoch+1}/{opt.EpochNum} — Avg Loss: {epoch_loss:.6f}")

# save model
os.makedirs(opt.ModelRoot, exist_ok=True)
out = os.path.join(opt.ModelRoot, opt.OutputFile)
torch.save(model.state_dict(), out)
print(f"💾 Model saved to {out}")

# optional plot
if opt.PlotGraph:
    plt.plot(range(1, opt.EpochNum + 1), loss_per_epoch, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    p = os.path.join(opt.ModelRoot, 'loss_curve.png')
    plt.savefig(p)
    print(f"📈 Loss curve → {p}")
