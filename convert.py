import os
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import functional as F
import cv2
from scipy.signal import convolve2d
from scipy.stats import mode
from tqdm import tqdm

# Downscaling functions
def downscale_labels_with_majority_voting(labels, scale_factor):
    if scale_factor not in [0.5, 0.25]:
        raise ValueError("Only scale factors 0.5 and 0.25 are supported.")

    kernel_size = int(1 / scale_factor)
    H, W = labels.shape
    H_new, W_new = H // kernel_size, W // kernel_size

    labels = labels.astype(float)
    labels[labels == 0] = np.nan

    reshaped = labels[:H_new * kernel_size, :W_new * kernel_size].reshape(H_new, kernel_size, W_new, kernel_size)
    blocks = reshaped.transpose(0, 2, 1, 3).reshape(H_new, W_new, -1)

    modes, _ = mode(blocks, axis=-1, nan_policy='omit')
    return np.nan_to_num(modes.squeeze(), nan=0).astype(np.int32)

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

# Cityscapes label mapping
CITYSCAPES_LABEL_MAP = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 1, 8: 2, 9: 0, 10: 0, 
                        11: 3, 12: 4, 13: 5, 14: 0, 15: 0, 16: 0, 17: 6, 18: 0, 19: 7, 20: 8, 
                        21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16, 29: 0, 
                        30: 0, 31: 17, 32: 18, 33: 19, -1: 0}

def process_and_save_dataset(base_dir, split, scale_factor=0.25):
    rgb_dir = os.path.join(base_dir, "leftImg8bit", split)
    disparity_dir = os.path.join(base_dir, "disparity", split)
    label_dir = os.path.join(base_dir, "gtFine", split)

    for root, _, files in os.walk(rgb_dir):
        for file in tqdm(files, desc=f"Processing: {root}"):
            if file.endswith("_leftImg8bit.png"):
                city = os.path.basename(root)
                rgb_file = os.path.join(root, file)
                disparity_file = os.path.join(disparity_dir, city, file.replace("_leftImg8bit.png", "_disparity.png"))
                label_file = os.path.join(label_dir, city, file.replace("_leftImg8bit.png", "_gtFine_labelIds.png"))

                if os.path.exists(disparity_file) and os.path.exists(label_file):
                    # Load images
                    rgb_np = np.array(Image.open(rgb_file).convert("RGB"))
                    disparity_np = np.array(Image.open(disparity_file)).astype(np.float32) / 65536.0
                    label_np = np.array(Image.open(label_file), dtype=np.int64)

                    # Convert RGB to YUV
                    yuv_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2YUV)
                    yuv_tensor = F.to_tensor(yuv_np).float()
                    disparity_tensor = torch.from_numpy(disparity_np).unsqueeze(0)

                    # Map labels
                    label_mapped = np.vectorize(CITYSCAPES_LABEL_MAP.get)(label_np)

                    # Combine YUV and disparity
                    yuvd_tensor = torch.cat([yuv_tensor, disparity_tensor], dim=0)
                    yuvd_np = yuvd_tensor.numpy()

                    # Downscale
                    yuvd_downscaled = downscale_exclude_invalid_multichannel(yuvd_np, scale_factor)
                    label_downscaled = downscale_labels_with_majority_voting(label_mapped, scale_factor)

                    # Save files
                    new_filename = file.replace("_leftImg8bit.png", "_ready4.npy")
                    np.save(os.path.join(root, new_filename), yuvd_downscaled.astype(np.float16))

                    label_filename = file.replace("_leftImg8bit.png", "_gtFine_ready4.npy")
                    np.save(os.path.join(label_dir, city, label_filename), label_downscaled.astype(np.uint8))

                    #print(f"Saved: {new_filename}, {label_filename}")

# Run the processing
if __name__=='__main__':
    base_dir = "/arf/home/myadiyaman/projeler/hydracity/data/cityscapes"
    #process_and_save_dataset(base_dir, "train", scale_factor=0.25)
    process_and_save_dataset(base_dir, "val", scale_factor=0.25)
    process_and_save_dataset(base_dir, "test", scale_factor=0.25)
