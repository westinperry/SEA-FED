from __future__ import absolute_import, print_function
import os
import numpy as np
import torch
import random

def mkdir(path):
    """Create a directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def crop_image(img, s):
    """Crop s pixels from all sides of the image."""
    if s > 0:
        if img.ndim == 3:  # C x H x W
            return img[:, s:-s, s:-s]
        elif img.ndim == 4:  # F x C x H x W
            return img[:, :, s:-s, s:-s]
        elif img.ndim == 5:  # N x F x C x H x W
            return img[:, :, :, s:-s, s:-s]
    return img  # return original if no cropping or unexpected shape

def tensor2numpy(tensor_in):
    """Convert PyTorch tensor to NumPy array."""
    return tensor_in.detach().cpu().numpy()

def get_subdir_list(path, is_sort=True):
    """Return a list of subdirectories in a given path."""
    subdirs = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    if is_sort:
        subdirs.sort()
    return subdirs

def get_file_list(path, is_sort=True):
    """Return a list of files in a given path."""
    files = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))]
    if is_sort:
        files.sort()
    return files

class UnNormalize(object):
    """Undo normalization on tensors using given mean and std."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor_in):
        t_out = tensor_in.clone()
        s = t_out.shape
        if len(s) == 5:  # N x C x F x H x W
            for i in range(s[1]):
                t_out[:, i, :, :, :] = t_out[:, i, :, :, :] * self.std[i] + self.mean[i]
        elif len(s) == 4:  # C x F x H x W
            for i in range(s[0]):
                t_out[i, :, :, :] = t_out[i, :, :, :] * self.std[i] + self.mean[i]
        return t_out

def vframes2imgs(frames_in, step=1, batch_idx=0):
    """
    Convert a batch of video frames (tensor) into numpy frames.
    Supports both 2D (N x F x C x H x W) and 3D (N x C x F x H x W) formats.
    """
    frames_np = tensor2numpy(frames_in)
    s = frames_np.shape

    if len(s) == 4:  # N x F x H x W (after squeeze)
        frames_np = frames_np[batch_idx]
        return frames_np[::step] if step > 1 else frames_np

    elif len(s) == 5:  # N x C x F x H x W
        frames_np = frames_np[batch_idx]
        frames_np = np.transpose(frames_np, (1, 0, 2, 3))  # F x C x H x W
        return frames_np[::step] if step > 1 else frames_np

def btv2btf(frames_in):
    """Convert tensor from shape (N x C x F x H x W) to (N x F x C x H x W)."""
    frames_np = tensor2numpy(frames_in)
    if frames_np.ndim == 5:
        return np.transpose(frames_np, (0, 2, 1, 3, 4))
    return frames_np

def get_model_setting(opt):
    return f"{opt.Dataset}"

def weights_init(m):
    """Initialize model weights for Conv and BatchNorm layers."""
    classname = m.__class__.__name__
    if 'Conv' in classname and hasattr(m, 'weight'):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname and hasattr(m, 'weight'):
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias'):
            torch.nn.init.constant_(m.bias.data, 0)

def seed(seed_val):
    """Set random seed for reproducibility."""
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    random.seed(seed_val)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_val)
