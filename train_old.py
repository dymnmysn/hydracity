import sys
import yaml
from HydraSalsa import HydraSalsa
sys.path.append('/ari/users/ibaskaya/projeler/hydrasalsa/utils')
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import OrderedDict
from glob import glob
from scipy.ndimage import convolve
import os

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from fastfill import FastFill
from scale3d import RandomRescaleRangeImage
from dskittiwaymo import SegmentationDataset
from combinedloader import MixedDataLoader

from metric_miou import calculate_classwise_intersection_union,calculate_final_miou_from_batches, calculate_miou
from printiou import print_miou_kitti, print_miou_waymo
from lovasz import Lovasz_softmax

from mappings import kitti_normalized_frequencies, waymovalidfreqs


if __name__=='__main__':
    
    num_classes_kitti = 20
    num_classes_waymo = 23
    inchannels = 5

    frequencies = kitti_normalized_frequencies
    frequencies_waymo = [i/sum(waymovalidfreqs) for i in waymovalidfreqs]
    max_epochs = 150               # number of epochs
    learning_rate = 0.01           # initial learning rate for SGD
    warmup_epochs = 1              # number of warmup epochs
    momentum = 0.9                 # momentum for SGD
    lr_decay = 0.99                # learning rate decay factor per epoch
    weight_decay = 0.0001          # weight decay for optimizer
    batch_size = 8                # batch size
    epsilon_w = 0.001 


    model = HydraSalsa([num_classes_kitti,num_classes_waymo],inchannels)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    print(device, ' is used')

    ffk = FastFill(tofill=0, indices=[0,1,2,3,4])
    ffw = FastFill(tofill=-1, indices=[0,1,2,3,4])
    transform_train = A.Compose([
        A.Resize(height=64, width=2048, interpolation=cv2.INTER_NEAREST, p=1),  # Resize
        A.ShiftScaleRotate(shift_limit=0.5, scale_limit=0.0, rotate_limit=0, 
                        border_mode=cv2.BORDER_WRAP, interpolation=cv2.INTER_NEAREST,
                        p=0.5),  
        A.RandomCrop(height = 64, width = 2048, p=1),
        #A.PadIfNeeded(min_height=64, min_width=2048, border_mode=0, value=0, mask_value=0),
        A.HorizontalFlip(p=0.5),  # Horizontal flip with 20% probability
        #A.CoarseDropout(max_holes=2, max_height=64, max_width=256, min_holes=1, min_height=1, min_width=1, fill_value=0, p=1),  # CoarseDropout instead of Cutout
        ToTensorV2()  # Convert to PyTorch tensors
    ], additional_targets={'mask': 'image'})
    transform_valid = A.Compose([
        A.Resize(height=64, width=2048, interpolation=cv2.INTER_NEAREST, p=1),  # Resize
        #A.RandomCrop(height = 64, width = 2048, p=1),
        #A.PadIfNeeded(min_height=64, min_width=2048, border_mode=0, value=0, mask_value=0),
        #A.HorizontalFlip(p=0.5),  # Horizontal flip with 20% probability
        #A.CoarseDropout(max_holes=2, max_height=64, max_width=256, min_holes=1, min_height=1, min_width=1, fill_value=0, p=1),  # CoarseDropout instead of Cutout
        ToTensorV2()  # Convert to PyTorch tensors
    ], additional_targets={'mask': 'image'})
    #pretransform = RandomRescaleRangeImage(p=1)
    pretransform = None

    ##Kitti
    train_dataset = SegmentationDataset(root = '/ari/users/ibaskaya/projeler/hydrasalsa/data/kitti', 
                                        split = 'training', transform=transform_train, 
                                        pretransform=pretransform, fastfill=ffk, iswaymo=False, width=2048)

    ktrain_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)

    validation_dataset = SegmentationDataset(root = '/ari/users/ibaskaya/projeler/hydrasalsa/data/kitti', 
                                        split = 'validation', transform=transform_valid, 
                                        pretransform=None, fastfill=ffk, iswaymo=False, width=2048)
    kvalidation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=2)

    ##Waymo
    wtrain_dataset = SegmentationDataset(root = '/ari/users/ibaskaya/projeler/hydrasalsa/data/waymo', 
                                        split = 'training', transform=transform_train, 
                                        pretransform=pretransform, fastfill=ffw, iswaymo=True, width=2650,unknown=-1)

    wtrain_dataloader = torch.utils.data.DataLoader(wtrain_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)

    wvalidation_dataset = SegmentationDataset(root = '/ari/users/ibaskaya/projeler/hydrasalsa/data/waymo', 
                                        split = 'validation', transform=transform_valid, 
                                        pretransform=None, fastfill=ffk, iswaymo=True, width=2650,unknown=-1)
    wvalidation_dataloader = torch.utils.data.DataLoader(wvalidation_dataset, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=2)

    #Mixed Loader
    train_dataloader = MixedDataLoader(ktrain_dataloader,wtrain_dataloader)

    inverse_frequencies = [1.0 / (f + epsilon_w) for f in frequencies]
    inverse_frequencies[0] = min(inverse_frequencies) / 10
    criterion_nll = nn.NLLLoss(weight=torch.tensor(inverse_frequencies).to(device))
    criterion_lovasz = Lovasz_softmax(ignore=0, from_logits=False)

    #For Waymo
    inverse_frequencies_waymo = [1.0 / (f + epsilon_w) for f in frequencies_waymo]
    inverse_frequencies_waymo[0] = min(inverse_frequencies_waymo) / 10
    criterion_nll_waymo = nn.NLLLoss(weight=torch.tensor(inverse_frequencies_waymo).to(device))
    criterion_lovasz_waymo = Lovasz_softmax(ignore=0, from_logits=False)

    # Model, optimizer, and scheduler setup
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    # Warmup scheduler for initial epochs
    def warmup_lr_scheduler(optimizer, warmup_epochs, initial_lr):
        def lr_lambda(epoch):
            return epoch / warmup_epochs if epoch < warmup_epochs else 1
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_epochs, learning_rate)

    # Training loop
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0

        # Learning rate scheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        for i, (batchk, batchw) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

            imagek, maskk = batchk
            imagew, maskw = batchw
            imagekw = torch.cat((imagek,imagew), dim=0)

            maskk = maskk.to(device)
            maskw = maskw.to(device)
            imagekw = imagekw.to(torch.float32).to(device)
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass and loss computation
            output1, output2 = model(imagekw)
            outputk, outputw = output1[:batch_size], output2[batch_size:]
        
            loss_kitti = criterion_nll(torch.log(outputk), maskk) + criterion_lovasz(outputk, maskk)
            loss_waymo = criterion_nll_waymo(torch.log(outputw), maskw) + criterion_lovasz_waymo(outputw, maskw)

            loss = 0.8*loss_kitti + 0.2*loss_waymo

            # Backpropagation and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{max_epochs}], Training Loss: {running_loss / len(train_dataloader):.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Validation phase
        model.eval()
        miou_total_kitti = 0.0
        miou_total_waymo = 0.0
        batch_results_kitti, batch_results_waymo = [], [] 

        with torch.no_grad():
            for i, (images, masks) in tqdm(enumerate(kvalidation_dataloader)):
                images, masks = images.to(torch.float32).to(device), masks.to(device)

                # Forward pass and predictions
                outputs = model(images)[0]
                _, preds = torch.max(outputs, dim=1)

                # mIoU calculation and class-wise IoU collection
                miou = calculate_miou(preds, masks, num_classes=num_classes_kitti, ignore_index=0)
                cwiou = calculate_classwise_intersection_union(preds, masks,num_classes=num_classes_kitti)
                batch_results_kitti.append(cwiou)

                miou_total_kitti += miou

            for i, (images, masks) in tqdm(enumerate(wvalidation_dataloader)):
                images, masks = images.to(torch.float32).to(device), masks.to(device)

                # Forward pass and predictions
                outputs = model(images)[1]
                _, preds = torch.max(outputs, dim=1)

                # mIoU calculation and class-wise IoU collection
                miou = calculate_miou(preds, masks, num_classes=num_classes_waymo, ignore_index=0)
                cwiou = calculate_classwise_intersection_union(preds, masks,num_classes=num_classes_waymo)
                batch_results_waymo.append(cwiou)

                miou_total_waymo += miou

        # Calculate and display mIoU metrics
        print('###################KITTI_START########################')
        classwise_iou, mean_iou, total_iou = calculate_final_miou_from_batches(batch_results_kitti, num_classes=num_classes_kitti)
        print_miou_kitti(classwise_iou, mean_iou, total_iou)
        avg_miou_kitti = miou_total_kitti / len(kvalidation_dataloader)
        print(f"Epoch [{epoch+1}/{max_epochs}], Validation mIoU Kitti: {avg_miou_kitti:.4f}")
        print('###################KITTI_END##########################')

        # Calculate and display mIoU metrics
        print('###################WAYMO_START########################')
        classwise_iou, mean_iou, total_iou = calculate_final_miou_from_batches(batch_results_waymo, num_classes=num_classes_waymo)
        print_miou_waymo(classwise_iou, mean_iou, total_iou)
        avg_miou_waymo = miou_total_waymo / len(wvalidation_dataloader)
        print(f"Epoch [{epoch+1}/{max_epochs}], Validation mIoU Waymo: {avg_miou_waymo:.4f}")
        print('###################WAYMO_END##########################')

        if epoch>20 and epoch%20==0:
            torch.save(model.state_dict(), f'model_state_dict_{epoch}.pth')

    torch.save(model.cpu().state_dict(), 'model_state_dict.pth')
