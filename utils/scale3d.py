import random
import torch
from torchvision import transforms

class RandomRescaleRangeImage:
    def __init__(self, scale_range=(1, 1.5), p=0.05, nan_pixel = -1):
        """
        Initializes the RandomRescaleRangeImage transform.
        
        Parameters:
        - scale_range: Tuple of (min_scale, max_scale) for random scaling.
        """
        self.scale_range = scale_range
        self.p = p
        self.nan_pixel = nan_pixel

    def __call__(self, image, label):
        """
        Applies random rescaling to both the image and segmentation label.

        Parameters:
        - image: The input image tensor of shape (C, H, W).
        - label: The input segmentation label tensor of shape (H, W).

        Returns:
        - rescaled_image: The rescaled image tensor.
        - rescaled_label: The rescaled segmentation label tensor.
        """
        if random.random() < self.p:
            return image, label

        # Randomly select a scaling factor
        scale = random.uniform(*self.scale_range)
        
        # Get original size
        original_size = image.shape[-2:]  # (H, W)
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        
        # Rescale the image dimensions (with interpolation)
        rescaled_image = transforms.Resize(new_size, interpolation=transforms.InterpolationMode.NEAREST)(torch.tensor(image))
        
        # Rescale the segmentation label (with nearest neighbor interpolation)
        rescaled_label = transforms.Resize(new_size, interpolation=transforms.InterpolationMode.NEAREST)(torch.tensor(label).unsqueeze(0))

        # Adjust the pixel values of the image based on the scale
        mask = rescaled_image != self.nan_pixel
        rescaled_image[mask] = rescaled_image[mask] / scale

        return rescaled_image.numpy(), rescaled_label.squeeze().numpy()