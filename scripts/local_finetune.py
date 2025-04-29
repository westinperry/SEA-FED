#!/usr/bin/env python3
import os
import argparse
import sys

# ensure project modules are on PYTHONPATH
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, '..')
sys.path.append(script_dir)
sys.path.append(project_root)

import data
import utils
from models.ae_3dconv import GatedAutoEncoderCov3D

def main():
    parser = argparse.ArgumentParser(description="Ditto personalization: local fine-tune of global model")
    parser.add_argument('--global-model', required=True,
                        help='Path to the FedAvg aggregated model checkpoint')
    parser.add_argument('--data-root',    required=True,
                        help='Root directory containing processed_{client} subfolder')
    parser.add_argument('--dataset',      required=True,
                        help='Dataset name (e.g. UCSD_P2_256)')
    parser.add_argument('--out-model',    required=True,
                        help='Output path for personalized model checkpoint')
    parser.add_argument('--lr',           type=float, default=1e-4,
                        help='Learning rate for personalization')
    parser.add_argument('--mu',           type=float, default=0.05,
                        help='Proximal regularization strength')
    parser.add_argument('--epochs',       type=int,   default=5,
                        help='Number of local fine-tuning epochs')
    parser.add_argument('--batch-size',   type=int,   default=8,
                        help='Batch size for fine-tuning')
    parser.add_argument('--num-workers',  type=int,   default=4,
                        help='Data loader workers')
    opt = parser.parse_args()

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load global model
    model = GatedAutoEncoderCov3D(chnum_in=1).to(device)
    model.load_state_dict(torch.load(opt.global_model, map_location=device))
    # snapshot for proximal term
    global_state = {k:v.clone().to(device) for k,v in model.state_dict().items()}
    model.train()

    # loss and optimizer
    loss_fn = nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    # data transforms and loader
    frame_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5],[0.5])
    ])
    idx_dir = os.path.join(opt.data_root, opt.dataset, 'Train_idx')
    frm_dir = os.path.join(opt.data_root, opt.dataset, 'Train')
    ds = data.VideoDataset(idx_dir, frm_dir, transform=frame_tf)
    loader = DataLoader(ds, batch_size=opt.batch_size,
                        shuffle=True, num_workers=opt.num_workers,
                        pin_memory=True)

    # fine-tuning loop
    for epoch in range(opt.epochs):
        for _, frames in loader:
            frames = frames.to(device)
            recon = model(frames)
            loss = loss_fn(recon, frames)
            # proximal to global snapshot
            prox = 0
            for name, p in model.named_parameters():
                prox += (p - global_state[name]).pow(2).sum()
            loss = loss + (opt.mu/2) * prox
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"➡️ Fine-tune Epoch {epoch+1}/{opt.epochs} — Loss {loss.item():.6f}")

    # save personalized model
    os.makedirs(os.path.dirname(opt.out_model), exist_ok=True)
    torch.save(model.state_dict(), opt.out_model)
    print(f"Personalized model saved to {opt.out_model}")

if __name__ == '__main__':
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from torch.utils.data import DataLoader
    main()
