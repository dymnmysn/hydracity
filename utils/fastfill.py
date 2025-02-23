import numpy as np
from scipy.ndimage import generic_filter, convolve
import random

def fast_interpolate(arr, thresholds=[5, 3], tofill = -1):
    mask = (arr == tofill)
    
    # Define a 3x3 kernel for counting neighbors
    kernel = np.ones((3, 3))
    kernel[1, 1] = 0  # Do not count the center pixel itself
    
    for threshold in thresholds:
        # Update the mask for -1 pixels
        mask = (arr == tofill)
        mmask = (~mask).astype(np.int8)
        
        # Count valid neighbors by applying convolution
        valid_neighbor_count = convolve(mmask, kernel, mode='constant', cval=0)

        # Compute the sum of valid neighbors
        neighbor_sum = convolve(np.where(mask, 0, arr), kernel, mode='constant', cval=0)

        # Find pixels that should be interpolated
        interpolate_mask = (valid_neighbor_count >= threshold) & mask

        # Interpolate those pixels by dividing the sum by the number of valid neighbors
        arr[interpolate_mask] = neighbor_sum[interpolate_mask] / valid_neighbor_count[interpolate_mask]

    return arr

class FastFill:
    def __init__(self, p=1, indices=[0,-1], thresholds=[5,3], tofill = -1, with_label=False, tofill_label = 0):
        self.p = p
        self.indices = indices
        self.with_label = with_label
        self.thresholds = thresholds
        self.tofill = tofill 
        self.tofill_label = tofill_label

    def __call__(self, image, label):
        if random.random() < self.p:
            return image, label

        for ind in self.indices:
            image[ind,...] = fast_interpolate(image[ind,...], thresholds=self.thresholds, tofill=self.tofill)

        if with_label:
            label = fast_interpolate(label, thresholds=self.thresholds, tofill=self.tofill_label)
        return image, label