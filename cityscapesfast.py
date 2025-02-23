import os
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as F
from torch.utils.data import Dataset
import cv2
from scipy.signal import convolve2d
from scipy.stats import mode


def downscale_labels_with_majority_voting(labels, scale_factor):
    if scale_factor not in [0.5, 0.25]:
        raise ValueError("Only scale factors 0.5 and 0.25 are supported.")
    
    kernel_size = int(1 / scale_factor)
    H, W = labels.shape
    H_new, W_new = H // kernel_size, W // kernel_size
    
    labels = labels.astype(float)
    labels[labels == 0] = np.nan
    
    reshaped = labels[:H_new * kernel_size, :W_new * kernel_size].reshape(
        H_new, kernel_size, W_new, kernel_size
    )
    blocks = reshaped.transpose(0, 2, 1, 3).reshape(H_new, W_new, -1)
    
    modes, _ = mode(blocks, axis=-1, nan_policy='omit')
    return np.nan_to_num(modes, nan=0).astype(int).squeeze()


def downscale_exclude_invalid_multichannel(image, scale_factor):
    if scale_factor not in [0.5, 0.25]:
        raise ValueError("Only scale factors 0.5 and 0.25 are supported.")
    
    kernel_size = int(1 / scale_factor)
    C, H, W = image.shape
    kernel = np.ones((kernel_size, kernel_size), dtype=float)
    H_new, W_new = H // kernel_size, W // kernel_size
    downscaled_image = np.zeros((C, H_new, W_new), dtype=float)
    
    for c in range(C):
        summed_values = convolve2d(image[c], kernel, mode='valid')[::kernel_size, ::kernel_size]
        valid_mask = image[c] > 0
        valid_counts = convolve2d(valid_mask.astype(float), kernel, mode='valid')[::kernel_size, ::kernel_size]
        downscaled_image[c] = np.divide(summed_values, valid_counts, out=np.zeros_like(summed_values), where=valid_counts > 0)
    
    return downscaled_image


class CityScapes(Dataset):
    def __init__(self, base_dir, split, transform=None):
        self.rgb_dir = os.path.join(base_dir, "leftImg8bit", split)
        self.disparity_dir = os.path.join(base_dir, "disparity", split)
        self.label_dir = os.path.join(base_dir, "gtFine", split)
        self.transform = transform
        self.samples = self._load_samples()
    
    def _load_samples(self):
        samples = []
        for root, _, files in os.walk(self.rgb_dir):
            for file in files:
                if file.endswith("_leftImg8bit.png"):
                    city = os.path.basename(root)
                    rgb_file = os.path.join(root, file)
                    disparity_file = os.path.join(self.disparity_dir, city, file.replace("_leftImg8bit.png", "_disparity.png"))
                    label_file = os.path.join(self.label_dir, city, file.replace("_leftImg8bit.png", "_gtFine_labelIds.png"))
                    if os.path.exists(disparity_file) and os.path.exists(label_file):
                        samples.append((rgb_file, disparity_file, label_file))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx, scale_factor=0.25):
        rgb_file, disparity_file, label_file = self.samples[idx]
        
        rgb_image = Image.open(rgb_file).convert("RGB")
        disparity_image = Image.open(disparity_file)
        label_image = Image.open(label_file)
        
        rgb_np = np.array(rgb_image)
        yuv_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2YUV)
        yuv_tensor = F.to_tensor(yuv_np).float()
        
        disparity_np = np.array(disparity_image).astype(np.float32) / 65536.0
        disparity_tensor = torch.from_numpy(disparity_np).unsqueeze(0)
        
        label_np = np.array(label_image, dtype=np.int64)
        label_tensor = torch.from_numpy(label_np).long()
        
        yuvd_tensor = torch.cat([yuv_tensor, disparity_tensor], dim=0)
        
        yuvd_np = yuvd_tensor.numpy()
        label_np = label_tensor.numpy()
        
        yuvd_downscaled = downscale_exclude_invalid_multichannel(yuvd_np, scale_factor)
        label_downscaled = downscale_labels_with_majority_voting(label_np, scale_factor)
        
        yuvd_tensor = torch.from_numpy(yuvd_downscaled).float()
        label_tensor = torch.from_numpy(label_downscaled).long()
        
        if self.transform:
            yuvd_tensor, label_tensor = self.transform(yuvd_tensor, label_tensor)
        
        return {"yuvd": yuvd_tensor, "label": label_tensor}
