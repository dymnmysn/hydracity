from mappings import sq_w2k, sq_k2w
from torch.utils.data import Dataset
import torch
import numpy as np
from glob import glob
import os


class SegmentationDataset(Dataset):
    def __init__(self, root = '/ari/users/ibaskaya/projeler/lidar-bonnetal/datasets/kitti', split = 'training', 
    transform=None, pretransform=None, fastfill=None, iswaymo=True, width=2656, unknown=0, xyz=False):
        
        imagepaths = sorted(glob(os.path.join(root,split,'images','*.npz')))
        labelpaths = [i.replace('images','labels') for i in imagepaths]
        self.datapaths  = [(imagepaths[i], labelpaths[i]) for i in range(len(imagepaths))]
        
        DEBUG = False
        if DEBUG:
            self.datapaths = self.datapaths[:2]
            
        self.transform = transform
        self.pretransform = pretransform
        self.fastfill = fastfill
        self.width = width
        
        """        self.ARCH = np.array([[16.7921, 15.1336],
                        [ 1.6576, 16.8768],
                        [ 0.8692, 14.5421],
                        [ 0.2838,  3.2056],
                        [-0.0222,  0.4402]])"""

        self.ARCH = np.array([[20.121, 13.786],
                        [ 1.74, 16.81],
                        [ 1.06, 14.05],
                        [ 0.288,  3.134],
                        [ 0.1443,  0.164]])


        if not iswaymo:
            self.ARCH = np.array([
                [12.12, 12.32],  
                [10.88, 11.47],  
                [0.23, 6.91],    
                [-1.04, 0.86],   
                [0.21, 0.16]     
            ])
            mapdict = sq_k2w
            max_index = max(mapdict.keys())  
            lookup_tensor = torch.zeros(max_index + 1, dtype=torch.long) 
            for old_idx, new_idx in mapdict.items():
                lookup_tensor[old_idx] = new_idx
            
            self.lookup_tensor = lookup_tensor
        else:
            self.lookup_tensor = None

        self.iswaymo = iswaymo

        self.shiftrange, self.scalerange = self.ARCH[0,0], self.ARCH[0,1]
        self.shiftintensity, self.scaleintensity = self.ARCH[4,0], self.ARCH[4,1]
        self.unknown = unknown

        self.xyz = xyz
        self.mx, self.sx = self.ARCH[1,0], self.ARCH[1,1]
        self.my, self.sy = self.ARCH[2,0], self.ARCH[2,1]
        self.mz, self.sz = self.ARCH[3,0], self.ARCH[3,1]

    
    def __len__(self):
        return len(self.datapaths)

    def __getitem__(self, idx):
        imagepath,labelpath = self.datapaths[idx]

        if not self.iswaymo:
            image, label = np.load(imagepath)['image'], np.load(labelpath)['label']
        else:
            image, label = np.load(imagepath)['array'], np.load(labelpath)['array']
            
        image, label = torch.tensor(image, dtype = torch.float32), torch.tensor(label, dtype=torch.long)
        
        if not self.iswaymo:
            label = self.lookup_tensor[label]

        image, label = image.numpy(), label.numpy()

        if self.fastfill:
            image,label = self.fastfill(image,label)

        if self.pretransform:
            image, label = self.pretransform(image, label)

        mask = image[0]!=self.unknown
        
        rangeimage = (image[0,...]-self.shiftrange)/self.scalerange
        intensityimage = (image[-1,...]-self.shiftintensity)/self.scaleintensity
        rangeimage = rangeimage*mask
        intensity = intensityimage*mask

        if self.xyz:
            ximage = ((image[1]-self.mx)/self.sx)*mask
            yimage = ((image[2]-self.my)/self.sy)*mask
            zimage = ((image[3]-self.mz)/self.sz)*mask
            image = np.stack((rangeimage,ximage,yimage,zimage,intensity))
        else:
            image = np.stack((rangeimage,intensity))

        if self.transform:
            augmented = self.transform(image=np.transpose(image, (1, 2, 0)), mask=label.astype(np.float32)[...,np.newaxis])
            image = augmented['image'].to(torch.float32)
            label = augmented['mask'].to(torch.long)

        return image, label.squeeze()

if __name__ == '__main__':
    sds = SegmentationDataset(root = '/truba/home/myadiyaman/Downloads/waymodataset/waymodataset', split = 'training', 
    transform=None, pretransform=None, fastfill=None, iswaymo=True, width=2656)
    sample = next(iter(sds))
    print('SIMPLE TEST')
    print(sample[0].shape,sample[1].shape)
    print('')
    from fastfill import FastFill
    ff = FastFill()
    sds = SegmentationDataset(root = '/truba/home/myadiyaman/Downloads/waymodataset/waymodataset', split = 'training', 
    transform=None, pretransform=None, fastfill=ff, iswaymo=True, width=2656)
    sample = next(iter(sds))
    print('WITH FASTFILL')
    print(sample[0].shape,sample[1].shape)
    print('')
    
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import cv2
    transform_train = A.Compose([
        A.Resize(height=64, width=2650, interpolation=cv2.INTER_NEAREST, p=1),  # Resize
        A.ShiftScaleRotate(shift_limit=0.5, scale_limit=0.0, rotate_limit=0, border_mode=cv2.BORDER_WRAP, p=1),  # Shift left with wrap-around effect
        A.RandomCrop(height = 64, width = 2650, p=1),
        A.PadIfNeeded(min_height=64, min_width=2656, border_mode=0, value=0, mask_value=0),
        A.HorizontalFlip(p=1),  # Horizontal flip with 20% probability
        A.CoarseDropout(max_holes=2, max_height=64, max_width=256, min_holes=1, min_height=1, min_width=1, fill_value=0, p=1),  # CoarseDropout instead of Cutout
        ToTensorV2()  # Convert to PyTorch tensors
    ], additional_targets={'mask': 'image'})  # Apply same augmentations to mask
    sds = SegmentationDataset(root = '/truba/home/myadiyaman/Downloads/waymodataset/waymodataset', split = 'training', 
    transform=transform_train, pretransform=None, fastfill=ff, iswaymo=True, width=2656)
    sample = next(iter(sds))
    print('WITH FASTFILL AND ALBUMENTATION')
    print(sample[0].shape,sample[1].shape)
    print('')

    from scale3d import RandomRescaleRangeImage
    pretransform = RandomRescaleRangeImage(p=1)
    sds = SegmentationDataset(root = '/truba/home/myadiyaman/Downloads/waymodataset/waymodataset', split = 'training', 
    transform=transform_train, pretransform=pretransform, fastfill=ff, iswaymo=True, width=2656)
    sample = next(iter(sds))
    print('WITH FASTFILL AND ALBUMENTATION AND SCALE')
    print(sample[0].shape,sample[1].shape)
    print('')