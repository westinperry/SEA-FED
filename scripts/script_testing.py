from __future__ import absolute_import, print_function
import os
import sys
import torch
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc

# Add project root to PYTHONPATH (assumes project root is one level above scripts)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import data
import utils
from models.ae_3dconv import AutoEncoderCov3D, GatedAutoEncoderCov3D
from options.testing_options import TestOptions

# === Parse Options ===
opt_parser = TestOptions()
opt = opt_parser.parse(is_print=True)
use_cuda = opt.UseCUDA
device = torch.device("cuda" if use_cuda else "cpu")

batch_size_in = opt.BatchSize
chnum_in_ = opt.ImgChnNum
framenum_in_ = opt.FrameNum
img_crop_size = 0

# === Paths ===
model_setting = utils.get_model_setting(opt)
data_root = os.path.join(opt.DataRoot, opt.Dataset)
data_frame_dir = os.path.join(data_root, 'Test')
data_idx_dir = os.path.join(data_root, 'Test_idx')
gt_dir = os.path.join(data_root, 'Test_gt')  # ground truth folder

model_root = opt.ModelRoot
model_path = opt.ModelFilePath if opt.ModelFilePath else os.path.join(model_root, model_setting + '.pt')

# Instead of using opt.OutRoot, define results folder one level above (project root)
project_root = os.path.join(os.path.dirname(__file__), '..')
results_folder = os.path.join(project_root, 'results')
utils.mkdir(results_folder)

# Create a subfolder for this experiment's results
te_res_root = results_folder
te_res_path = os.path.join(te_res_root, 'res_' + model_setting)
utils.mkdir(te_res_path)

# === Load Model ===
if opt.ModelName == 'AE':
    model = AutoEncoderCov3D(chnum_in_)
elif opt.ModelName == 'Gated_AE':
    model = GatedAutoEncoderCov3D(chnum_in_)
else:
    raise ValueError("Unknown ModelName")

model_para = torch.load(model_path)
model.load_state_dict(model_para)
model.to(device)
model.eval()

# === Transforms ===
norm_mean = [0.5] * chnum_in_
norm_std = [0.5] * chnum_in_

frame_trans = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std)
])
unorm_trans = utils.UnNormalize(mean=norm_mean, std=norm_std)

# === Get Video List ===
video_list = utils.get_subdir_list(data_idx_dir)
video_num = len(video_list)

# === Run Inference and Save Reconstruction Errors ===
with torch.no_grad():
    for ite_vid in range(video_num):
        video_name = video_list[ite_vid]
        video_idx_path = os.path.join(data_idx_dir, video_name)
        video_frame_path = os.path.join(data_frame_dir, video_name)
        idx_name_list = sorted([name for name in os.listdir(video_idx_path)
                                if os.path.isfile(os.path.join(video_idx_path, name))])

        print('[vidx %02d/%d] [vname %s]' % (ite_vid + 1, video_num, video_name))
        recon_error_list = []

        video_dataset = data.VideoDatasetOneDir(video_idx_path, video_frame_path, transform=frame_trans)
        video_data_loader = DataLoader(video_dataset, batch_size=batch_size_in, shuffle=False)

        for batch_idx, (item, frames) in enumerate(video_data_loader):
            idx_name = idx_name_list[item[0]]
            idx_data = sio.loadmat(os.path.join(video_idx_path, idx_name))
            frame_idx = idx_data['idx'][0, :]

            frames = frames.to(device)

            if opt.ModelName == 'AE' or opt.ModelName == 'Gated_AE':
                recon_frames = model(frames)
                recon_np = utils.vframes2imgs(unorm_trans(recon_frames.data), step=1, batch_idx=0)
                input_np = utils.vframes2imgs(unorm_trans(frames.data), step=1, batch_idx=0)
                r = utils.crop_image(recon_np, img_crop_size) - utils.crop_image(input_np, img_crop_size)
                recon_error = np.mean(r ** 2)
                recon_error_list.append(recon_error)

        np.save(os.path.join(te_res_path, video_name + '.npy'), recon_error_list)

# === Evaluation ===
utils.eval_video(data_root, te_res_path, is_show=False)

# === Plot ROC Curve & Save AUC ===
print("===> Calculating ROC Curve and AUC")

all_scores = []
all_labels = []

for video_name in video_list:
    pred_path = os.path.join(te_res_path, video_name + '.npy')
    gt_path = os.path.join(gt_dir, video_name + '.mat')  # Change to .mat

    if os.path.exists(pred_path) and os.path.exists(gt_path):
        scores = np.load(pred_path)
        labels = sio.loadmat(gt_path)['l'][0]  # Load .mat file, access 'l' key

        # Align with eval_video: trim labels to match sequence-level scores
        labels_res = labels[8:-7]

        # Ensure matching lengths
        if len(scores) != len(labels_res):
            print(f"[Warning] Mismatch in lengths for {video_name}: scores {len(scores)}, labels {len(labels_res)}")
            continue

        # Normalize scores as in eval_video
        scores_norm = scores - np.min(scores)
        scores_norm = 1 - scores_norm / np.max(scores_norm) if np.max(scores_norm) > 0 else scores_norm

        # Match label transformation in eval_video
        all_scores.extend(scores_norm)
        all_labels.extend(1 - labels_res + 1)  # 0->2 (normal), 1->1 (anomaly)
    else:
        print(f"[Warning] Missing prediction or GT for {video_name}")

all_scores = np.array(all_scores)
all_labels = np.array(all_labels)

if all_scores.size == 0 or all_labels.size == 0:
    print("No valid prediction/GT pairs found. Please check your prediction and ground truth paths.")
else:
    # Compute ROC and AUC, matching eval_video's pos_label=2
    fpr, tpr, _ = roc_curve(all_labels, all_scores, pos_label=2)
    roc_auc = auc(fpr, tpr)

    # Save ROC curve plot in a subfolder inside the results folder
    plot_save_dir = os.path.join(results_folder, 'plots')
    utils.mkdir(plot_save_dir)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.4f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.grid(True)

    plot_path = os.path.join(plot_save_dir, 'roc_auc_curve.png')
    plt.savefig(plot_path)
    plt.close()

    # Save AUC value to a text file in the results folder
    with open(os.path.join(results_folder, 'roc_auc_score.txt'), 'w') as f:
        f.write('AUC: {:.4f}\n'.format(roc_auc))

    print(f"ROC curve saved to {plot_path}")
    print(f"AUC score: {roc_auc:.4f}")

    # Append the AUC value to the results.txt file in the results folder
    results_txt_path = os.path.join(results_folder, 'results.txt')
    with open(results_txt_path, "a") as f:
        f.write(str(roc_auc) + "\n")
