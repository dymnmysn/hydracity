import numpy as np
from scipy.signal import convolve2d

def downscale_exclude_invalid(image, scale_factor=0.25):
    """
    Downscale an image using convolution while excluding invalid pixels (value 0).

    Args:
        image (np.ndarray): Input image with invalid pixels set to 0.
        scale_factor (float): Downscaling factor. Supported values are 0.5 and 0.25.

    Returns:
        np.ndarray: Downscaled image.
    """
    # Determine kernel size based on scale factor
    if scale_factor == 0.5:
        kernel_size = 2
    elif scale_factor == 0.25:
        kernel_size = 4
    else:
        raise ValueError("Only scale factors 0.5 and 0.25 are supported.")

    # Create the kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=float)

    # Sum of valid pixel values in each block
    summed_values = convolve2d(image, kernel, mode='valid')[::kernel_size, ::kernel_size]

    # Count of valid pixels in each block
    valid_mask = image > 0
    valid_counts = convolve2d(valid_mask.astype(float), kernel, mode='valid')[::kernel_size, ::kernel_size]

    # Compute the average only for valid pixels
    downscaled_image = np.divide(
        summed_values, valid_counts, 
        out=np.zeros_like(summed_values), 
        where=valid_counts > 0
    )

    return downscaled_image

def downscale_exclude_invalid_multichannel(image, scale_factor=0.25):
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
    
    # Initialize output shape
    H_new, W_new = H // kernel_size, W // kernel_size
    downscaled_image = np.zeros((C, H_new, W_new), dtype=float)
    
    for c in range(C):
        # Process each channel independently
        summed_values = convolve2d(image[c], kernel, mode='valid')[::kernel_size, ::kernel_size]
        valid_mask = image[c] > 0
        valid_counts = convolve2d(valid_mask.astype(float), kernel, mode='valid')[::kernel_size, ::kernel_size]
        
        downscaled_image[c] = np.divide(
            summed_values, valid_counts,
            out=np.zeros_like(summed_values),
            where=valid_counts > 0
        )
    
    return downscaled_image

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

