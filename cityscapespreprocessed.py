import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CityScapes(Dataset):

    def __init__(self, base_dir, split="train", transform=None):
        self.base_dir = base_dir
        self.split = split
        self.transform = transform

        self.image_dir = os.path.join(base_dir, "leftImg8bit", split)
        self.label_dir = os.path.join(base_dir, "gtFine", split)

        self.image_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(self.image_dir)
            for file in files if file.endswith("_ready4.npy")
        ]

        self.label_files = [
            os.path.join(self.label_dir, os.path.relpath(file, self.image_dir).replace("_ready4.npy", "_gtFine_ready4.npy"))
            for file in self.image_files
        ]


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = self.label_files[idx]

        # Load the npy files
        image = np.load(image_path).astype(np.float32)  # Shape: (4, H, W)  (YUV + disparity)
        label = np.load(label_path).astype(np.int64)    # Shape: (H, W)

        # Convert to PyTorch tensors
        yuvd_tensor = torch.from_numpy(image)
        label_tensor = torch.from_numpy(label)
        if self.transform:
            yuvd_tensor, label_tensor = self.transform(yuvd_tensor, label_tensor)

        return {"yuvd": yuvd_tensor, "label": label_tensor}