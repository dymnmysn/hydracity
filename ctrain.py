import os
import sys
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F
from tqdm import tqdm

# Add utils to path
sys.path.append('/arf/home/myadiyaman/projeler/hydracity/utils')

# Project-specific imports
from SalsaCity import SalsaNext
from lovasz import Lovasz_softmax
from iou_eval import iouEval
from cityscapespreprocessed import CityScapes



# ---------------------------
# Training and Validation
# ---------------------------
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0

    # Adjust learning rate scheduler
    if epoch < WARMUP_EPOCHS:
        warmup_scheduler.step()
    else:
        scheduler.step()

    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
        images = batch['yuvd'].float().to(device)
        masks  = batch['label'].long().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion_nll(torch.log(outputs), masks) + criterion_lovasz(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if DEBUG:
            break

    avg_loss = running_loss / len(train_dataloader)
    lr_current = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] Loss: {avg_loss:.4f} | LR: {lr_current:.6f}")

def validate(epoch):
    model.eval()
    metric.reset()
    with torch.no_grad():
        for batch in tqdm(validation_dataloader, desc=f"Validation Epoch {epoch+1}"):
            images = batch['yuvd'].float().to(device)
            masks  = batch['label'].long().to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            metric.addBatch(preds, masks)
            if DEBUG:
                break
    mean_iou = metric.getIoU()[0].item()
    acc = metric.getacc().item()
    print(f"Epoch [{epoch+1}/{MAX_EPOCHS}] Validation mIoU: {mean_iou:.4f} | Acc: {acc:.4f}")
    return mean_iou

if __name__=='__main__':
    # ---------------------------
    # Datasets and DataLoaders
    # ---------------------------
    cityscapes_path = "/arf/home/myadiyaman/projeler/hydracity/data/cityscapes"
    train_dataset = CityScapes(cityscapes_path, split="train")
    val_dataset = CityScapes(cityscapes_path, split="val")

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    validation_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # ---------------------------
    # Hyperparameters and Settings
    # ---------------------------
    DEBUG = False
    #DEBUG = True
    NUM_CLASSES    = 20
    MAX_EPOCHS     = 150
    LEARNING_RATE  = 0.01
    WARMUP_EPOCHS  = 1
    MOMENTUM       = 0.9
    LR_DECAY       = 0.99
    WEIGHT_DECAY   = 0.0001
    EPSILON_W      = 0.001

    # Class frequencies for loss weighting
    FREQUENCIES = [
    0.10756825, 0.32920191, 0.05440517, 0.20442969, 0.0058648,  0.00783859,
    0.01110949, 0.00184233, 0.00490152, 0.14157578, 0.01028896, 0.03579445,
    0.01084,    0.00120617, 0.06207226, 0.00236596, 0.00207926, 0.00206316,
    0.00087622, 0.00367605]

    # ---------------------------
    # Model Setup
    # ---------------------------
    model = SalsaNext(NUM_CLASSES, 4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('The device is', device)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # ---------------------------
    # Loss Functions
    # ---------------------------
    inverse_frequencies = [1.0 / (f + EPSILON_W) for f in FREQUENCIES]
    inverse_frequencies[0] = min(inverse_frequencies) / 10  # Adjust background weight
    criterion_nll    = nn.NLLLoss(weight=torch.tensor(inverse_frequencies).to(device))
    criterion_lovasz = Lovasz_softmax(ignore=0, from_logits=False)

    # ---------------------------
    # Optimizer and Schedulers
    # ---------------------------
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scheduler = ExponentialLR(optimizer, gamma=LR_DECAY)
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: epoch / WARMUP_EPOCHS if epoch < WARMUP_EPOCHS else 1)

    # ---------------------------
    # Metric
    # ---------------------------
    metric = iouEval(NUM_CLASSES, device, 0)
    best_mean_iou = 0.0

    # ---------------------------
    # Main Training Loop
    # ---------------------------
    for epoch in range(MAX_EPOCHS):
        train_one_epoch(epoch)
        mean_iou = validate(epoch)

        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            torch.save(model.state_dict(), 'kitti_best_miou_checkpoint.pth')
            print(f"New best mIoU: {best_mean_iou:.4f} - model saved.")

    torch.save(model.cpu().state_dict(), 'waymo_last_checkpoint.pth')
