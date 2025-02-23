import os
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import cv2

import numpy as np
from scipy.signal import convolve2d

from scipy.stats import mode

def downscale_labels_with_majority_voting(labels, scale_factor):
    """
    Downscale a label image using majority voting, excluding invalid pixels (value 0),
    using NaN handling.

    Args:
        labels (np.ndarray): Input label image of shape (H, W).
        scale_factor (float): Downscaling factor. Supported values are 0.5 and 0.25.

    Returns:
        np.ndarray: Downscaled label image of shape (H_new, W_new).
    """
    if scale_factor == 0.5:
        kernel_size = 2
    elif scale_factor == 0.25:
        kernel_size = 4
    else:
        raise ValueError("Only scale factors 0.5 and 0.25 are supported.")

    H, W = labels.shape
    kernel_size = int(1 / scale_factor)
    H_new, W_new = H // kernel_size, W // kernel_size

    # Convert invalid pixels (0) to NaN
    labels = labels.astype(float)
    labels[labels == 0] = np.nan

    # Reshape into blocks
    reshaped = labels[:H_new * kernel_size, :W_new * kernel_size].reshape(
        H_new, kernel_size, W_new, kernel_size
    )

    # Rearrange to group corresponding blocks
    blocks = reshaped.transpose(0, 2, 1, 3).reshape(H_new, W_new, -1)

    # Compute mode while ignoring NaN
    downscaled_labels = np.zeros((H_new, W_new), dtype=int)
    modes, _ = mode(blocks, axis=-1, nan_policy='omit')
    # Replace NaN in modes with 0, then cast to int
    modes = np.nan_to_num(modes, nan=0).astype(int)

    return modes.squeeze()

def downscale_exclude_invalid_multichannel(image, scale_factor):
    """
    Downscale a multi-channel image using convolution while excluding invalid pixels (value 0).

    Args:
        image (np.ndarray): Input image of shape (C, H, W) with invalid pixels set to 0.
        scale_factor (float): Downscaling factor. Supported values are 0.5 and 0.25.

    Returns:
        np.ndarray: Downscaled image with shape (C, H_new, W_new).
    """
    if scale_factor == 0.5:
        kernel_size = 2
    elif scale_factor == 0.25:
        kernel_size = 4
    else:
        raise ValueError("Only scale factors 0.5 and 0.25 are supported.")
    
    C, H, W = image.shape
    kernel = np.ones((kernel_size, kernel_size), dtype=float)
    
    H_new, W_new = H // kernel_size, W // kernel_size
    downscaled_image = np.zeros((C, H_new, W_new), dtype=float)
    
    for c in range(C):
        summed_values = convolve2d(image[c], kernel, mode='valid')[::kernel_size, ::kernel_size]
        valid_mask = image[c] > 0
        valid_counts = convolve2d(valid_mask.astype(float), kernel, mode='valid')[::kernel_size, ::kernel_size]
        
        downscaled_image[c] = np.divide(
            summed_values, valid_counts,
            out=np.zeros_like(summed_values),
            where=valid_counts > 0
        )
    
    return downscaled_image

CITYSCAPES_LABEL_MAP = {
    0: 0,      # Unlabeled -> Ignore (previously 255)
    1: 0,      # Ego vehicle -> Ignore
    2: 0,      # Rectification border -> Ignore
    3: 0,      # Out of ROI -> Ignore
    4: 0,      # Static -> Ignore
    5: 0,      # Dynamic -> Ignore
    6: 0,      # Ground -> Ignore
    7: 1,      # Road (0 -> 1)
    8: 2,      # Sidewalk (1 -> 2)
    9: 0,      # Parking -> Ignore
    10: 0,     # Rail track -> Ignore
    11: 3,     # Building (2 -> 3)
    12: 4,     # Wall (3 -> 4)
    13: 5,     # Fence (4 -> 5)
    14: 0,     # Guard rail -> Ignore
    15: 0,     # Bridge -> Ignore
    16: 0,     # Tunnel -> Ignore
    17: 6,     # Pole (5 -> 6)
    18: 0,     # Polegroup -> Ignore
    19: 7,     # Traffic light (6 -> 7)
    20: 8,     # Traffic sign (7 -> 8)
    21: 9,     # Vegetation (8 -> 9)
    22: 10,    # Terrain (9 -> 10)
    23: 11,    # Sky (10 -> 11)
    24: 12,    # Person (11 -> 12)
    25: 13,    # Rider (12 -> 13)
    26: 14,    # Car (13 -> 14)
    27: 15,    # Truck (14 -> 15)
    28: 16,    # Bus (15 -> 16)
    29: 0,     # Caravan -> Ignore
    30: 0,     # Trailer -> Ignore
    31: 17,    # Train (16 -> 17)
    32: 18,    # Motorcycle (17 -> 18)
    33: 19,    # Bicycle (18 -> 19)
    -1: 0      # License plate -> Ignore
}
    

class CityScapes(Dataset): #YUVD dataset
    def __init__(self, base_dir, split, transform=None):
        """
        Initializes the dataset with RGB, disparity, and segmentation label directories.

        Args:
            base_dir (str): Base directory of the Cityscapes dataset.
            split (str): Dataset split - 'train', 'val', or 'test'.
            transform (callable, optional): Transform to apply to the images.
        """
        self.rgb_dir = os.path.join(base_dir, "leftImg8bit", split)
        self.disparity_dir = os.path.join(base_dir, "disparity", split)
        self.label_dir = os.path.join(base_dir, "gtFine", split)
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        """Loads all valid RGB, disparity, and segmentation label file pairs."""
        samples = []
        for root, _, files in os.walk(self.rgb_dir):
            for file in files:
                if file.endswith("_leftImg8bit.png"):
                    city = os.path.basename(root)
                    rgb_file = os.path.join(root, file)
                    disparity_file = os.path.join(
                        self.disparity_dir,
                        city,
                        file.replace("_leftImg8bit.png", "_disparity.png")
                    )
                    label_file = os.path.join(
                        self.label_dir,
                        city,
                        file.replace("_leftImg8bit.png", "_gtFine_labelIds.png")
                    )
                    if os.path.exists(disparity_file) and os.path.exists(label_file):
                        samples.append((rgb_file, disparity_file, label_file))
        return samples

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx, scale_factor=0.25):
        """
        Retrieves the RGB, disparity, and segmentation label at the specified index,
        with optional downscaling.

        Args:
            idx (int): Index of the sample.
            scale_factor (float): Downscaling factor (default: 0.5).

        Returns:
            dict: A dictionary with 'yuvd' and 'label' keys.
        """
        rgb_file, disparity_file, label_file = self.samples[idx]

        # Read RGB, disparity, and label images
        rgb_image = Image.open(rgb_file).convert("RGB")
        disparity_image = Image.open(disparity_file)
        label_image = Image.open(label_file)

        # Convert RGB to YUV
        rgb_np = np.array(rgb_image)
        yuv_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2YUV)
        yuv_tensor = F.to_tensor(yuv_np).float()

        # Convert disparity to tensor
        disparity_np = np.array(disparity_image).astype(np.float32) / 65536.0
        disparity_tensor = torch.from_numpy(disparity_np).unsqueeze(0)

        # Convert label to tensor
        label_np = np.array(label_image, dtype=np.int64)
        label_mapped = np.vectorize(CITYSCAPES_LABEL_MAP.get)(label_np)
        label_tensor = torch.from_numpy(label_mapped).long()

        # Combine YUV and disparity into a tensor
        yuvd_tensor = torch.cat([yuv_tensor, disparity_tensor], dim=0)

        # Apply downscaling
        yuvd_np = yuvd_tensor.numpy()
        label_np = label_tensor.numpy()

        yuvd_downscaled = downscale_exclude_invalid_multichannel(yuvd_np, scale_factor)
        label_downscaled = downscale_labels_with_majority_voting(label_np, scale_factor)

        yuvd_tensor = torch.from_numpy(yuvd_downscaled).float()
        label_tensor = torch.from_numpy(label_downscaled).long()

        if self.transform:
            yuvd_tensor, label_tensor = self.transform(yuvd_tensor, label_tensor)

        return {"yuvd": yuvd_tensor, "label": label_tensor}