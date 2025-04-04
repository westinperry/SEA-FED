from __future__ import print_function, absolute_import
import torch
from torch.utils.data import Dataset
import os, os.path
import scipy.io as sio
from skimage import io
from torchvision import transforms
import numpy as np


# Video index files are organized in correlated sub folders.
# N x C x T x H x W
import torch
from torch.utils.data import Dataset
import os
import scipy.io as sio
from skimage import io

class VideoDataset(Dataset):
    """Dataset for training with video clips from subdirectories."""
    def __init__(self, idx_root, frame_root, transform):
        self.idx_root = idx_root
        self.frame_root = frame_root
        
        # Get sorted list of video subdirectories
        self.video_list = sorted([name for name in os.listdir(idx_root) 
                                 if os.path.isdir(os.path.join(idx_root, name))])
        
        # Build list of all idx file paths
        self.idx_path_list = []
        for video_name in self.video_list:
            video_idx_dir = os.path.join(idx_root, video_name)
            idx_files = sorted([name for name in os.listdir(video_idx_dir) 
                               if os.path.isfile(os.path.join(video_idx_dir, name))])
            self.idx_path_list.extend(os.path.join(video_idx_dir, fname) for fname in idx_files)
        
        self.transform = transform

    def __len__(self):
        return len(self.idx_path_list)

    def __getitem__(self, item):
        """Return a video clip as a tensor of stacked frames."""
        idx_path = self.idx_path_list[item]
        idx_data = sio.loadmat(idx_path)
        v_name = idx_data['v_name'][0]
        frame_idx = idx_data['idx'][0, :]
        
        v_dir = os.path.join(self.frame_root, v_name)
        frames = torch.stack([self.transform(io.imread(os.path.join(v_dir, f'{i:03d}.jpg'))) 
                             for i in frame_idx], dim=1)
        
        return item, frames


class VideoDatasetOneDir(Dataset):
    """Dataset for testing with video clips from a single directory."""
    def __init__(self, idx_dir, frame_root, transform):
        self.idx_dir = idx_dir
        self.frame_root = frame_root
        
        # Get sorted list of idx files
        self.idx_name_list = sorted([name for name in os.listdir(idx_dir) 
                                    if os.path.isfile(os.path.join(idx_dir, name))])
        
        self.transform = transform

    def __len__(self):
        return len(self.idx_name_list)

    def __getitem__(self, item):
        """Return a video clip as a tensor of stacked frames."""
        idx_name = self.idx_name_list[item]
        idx_data = sio.loadmat(os.path.join(self.idx_dir, idx_name))
        v_name = idx_data['v_name'][0]
        frame_idx = idx_data['idx'][0, :]
        
        v_dir = self.frame_root  # Frames are in frame_root directly
        frames = torch.stack([self.transform(io.imread(os.path.join(v_dir, f'{i:03d}.jpg'))) 
                             for i in frame_idx], dim=1)
        
        return item, frames