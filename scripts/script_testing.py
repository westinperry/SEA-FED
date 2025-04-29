# script_testing.py

from __future__ import print_function
import os
import sys
import re
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc

# set up paths
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(script_dir, "..")
sys.path.append(project_root)

import data, utils
from models.ae_3dconv import AutoEncoderCov3D, GatedAutoEncoderCov3D
from options.testing_options import TestOptions

# parse args
opt = TestOptions().parse(is_print=True)
device = torch.device("cuda" if opt.UseCUDA else "cpu")

# determine model path
model_path = opt.ModelFilePath or os.path.join(
    opt.ModelRoot,
    utils.get_model_setting(opt) + f"_r{opt.Round}.pt"
)
setting = utils.get_model_setting(opt)
# infer round_num: prefer opt.Round, else parse from filename
if opt.Round is not None:
    round_num = opt.Round
else:
    basename = os.path.basename(model_path)
    m = re.search(r'(\d+)(?=\.pt$)', basename)
    if m:
        round_num = int(m.group(1))
    else:
        raise ValueError(f"Cannot infer round number from '{basename}'")

# infer client ID from DataRoot (e.g. "../datasets/processed_1")
client_id = os.path.basename(opt.DataRoot).split("_")[-1]

# make top‑level results/
results_root = os.path.join(project_root, "results")
utils.mkdir(results_root)

# make per‑client graph folder
graph_dir = os.path.abspath(
    os.path.join(results_root,
                 f"client_{client_id}",
                 f"graph_round{round_num}")
)
utils.mkdir(graph_dir)

# folder for recon errors
recon_root = os.path.abspath(
    os.path.join(results_root,
                 "recon_errors",
                 f"{setting}_r{round_num}")
)
utils.mkdir(recon_root)

# load model
if opt.ModelName == "AE":
    model = AutoEncoderCov3D(opt.ImgChnNum)
elif opt.ModelName == "Gated_AE":
    model = GatedAutoEncoderCov3D(opt.ImgChnNum)
else:
    raise ValueError("Unknown ModelName")

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device).eval()

# transforms
norm      = [0.5] * opt.ImgChnNum
frame_tf  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm, norm)
])
unorm_tf  = utils.UnNormalize(mean=norm, std=norm)

# data dirs
idx_dir = os.path.join(opt.DataRoot, opt.Dataset, "Test_idx")
frm_dir = os.path.join(opt.DataRoot, opt.Dataset, "Test")
gt_dir  = os.path.join(opt.DataRoot, opt.Dataset, "Test_gt")
videos  = utils.get_subdir_list(idx_dir)

# inference → save per‑video recon errors
with torch.no_grad():
    for vid in videos:
        print(f"▶ {vid}")
        mat_files = sorted(
            f for f in os.listdir(os.path.join(idx_dir, vid))
            if f.endswith(".mat")
        )
        errors = []
        loader = DataLoader(
            data.VideoDatasetOneDir(
                os.path.join(idx_dir, vid),
                os.path.join(frm_dir, vid),
                transform=frame_tf
            ),
            batch_size=opt.BatchSize,
            shuffle=False
        )
        for item, frames in loader:
            mat = sio.loadmat(
                os.path.join(idx_dir, vid, mat_files[item[0]])
            )['idx'][0]
            frames = frames.to(device)
            recon  = model(frames)
            r_np   = utils.vframes2imgs(unorm_tf(recon.data),
                                        batch_idx=0, step=1)
            i_np   = utils.vframes2imgs(unorm_tf(frames.data),
                                        batch_idx=0, step=1)
            diff   = utils.crop_image(r_np, 0) - utils.crop_image(i_np, 0)
            errors.append(np.mean(diff ** 2))
        np.save(os.path.join(recon_root, f"{vid}.npy"), errors)
print(f"Saved recon errors → {recon_root}")

# ROC & AUC → plot into per‑client graph folder
scores, labels = [], []
for vid in videos:
    sc = np.load(os.path.join(recon_root, f"{vid}.npy"))
    lg = sio.loadmat(os.path.join(gt_dir, f"{vid}.mat"))['l'][0][8:-7]
    if len(sc) != len(lg):
        print(f"[!] length mismatch {vid}")
        continue
    sn = sc - sc.min()
    sn = 1 - sn/sn.max() if sn.max() > 0 else sn
    scores.extend(sn)
    labels.extend(1 - lg + 1)

if scores and labels:
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=2)
    my_auc      = auc(fpr, tpr)

    # save ROC plot
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=f"AUC={my_auc:.4f}")
    plt.plot([0, 1], [0, 1], lw=1, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Round {round_num} — Client {client_id}")
    plt.grid(True)

    roc_png = os.path.join(
        graph_dir,
        f"roc_client{client_id}_r{round_num}.png"
    )
    plt.savefig(roc_png)
    plt.close()
    print(f"Saved ROC → {roc_png}")

    # write AUC text and append summary
    auc_txt = os.path.join(
        graph_dir,
        f"auc_client{client_id}_r{round_num}.txt"
    )
    with open(auc_txt, "w") as f:
        f.write(f"Round {round_num}, Client {client_id}: AUC={my_auc:.4f}\n")

    summary = os.path.join(results_root, "results.txt")
    with open(summary, "a") as f:
        f.write(f"{my_auc:.8f}\n")
    print(f"Appended AUC → {summary}")
else:
    print("No valid scores/labels for ROC/AUC")
