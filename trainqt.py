import os
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
from glob import glob



from lovasz import Lovasz_softmax
from parser1 import Parser
from mappings import kitti, kitti_normalized_frequencies ,kitticolormap,sem2sem, kitti_lm_inv,waymocolormap,waymo,sensor_kitti,sensor_waymo,waymo_lmap, waymo_lmap_inv,waymovalidfreqs
from iou_eval import iouEval
from combinedloader import KittiWaymoTrainLoader

if __name__=='__main__':

    DEBUG = False

    num_classes = 23
    batch_number = 32
    batch_size,workers = 32, 24
    max_epochs = 40             
    learning_rate = 0.002     
    warmup_epochs = 1            
    momentum = 0.9              
    lr_decay = 0.99                
    weight_decay = 0.0001        
    batch_size = 8                
    epsilon_w = 0.001 

    freqsum = sum(waymovalidfreqs)
    frequencies = [i/freqsum for i in waymovalidfreqs]

    root_waymo = '/ari/users/ibaskaya/projeler/hydrasalsa/data/waymo'
    root_kitti = '/ari/users/ibaskaya/projeler/hydrasalsa/data/kittiorig'

    if DEBUG:
        root_kitti = '/ari/users/ibaskaya/projeler/hydrasalsa/data/kittismall'

    
    
    train_sequences = [0,1,2,3,4,5,6,7,9,10]
    valid_sequences = [8]
    test_sequences = None
    labels_waymo = waymo
    color_map_waymo = waymocolormap
    learning_map_waymo = waymo_lmap
    learning_map_inv_waymo = waymo_lmap_inv

    labels_kitti = kitti
    color_map_kitti = kitticolormap
    learning_map_kitti = sem2sem
    learning_map_inv_kitti = kitti_lm_inv

    
    max_points_waymo=170000
    max_points_kitti = 150000
    gt=True
    transform=False
    iswaymo = False
    istrain = False

    float_model = HydraSalsa(nclasses=[20,23], inchannels=5)
    state_dict = torch.load('/ari/users/ibaskaya/projeler/hydrasalsa/best_kitti_state_dict.pth')#, map_location=torch.device('cpu'))
    float_model.load_state_dict(state_dict)
    original_state_dict = float_model.state_dict()
    float_model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
    model = torch.quantization.prepare_qat(float_model)

    parser_kitti = Parser(root_kitti,              # directory for data
               train_sequences,   # sequences to train
               valid_sequences,   # sequences to validate.
               test_sequences,    # sequences to test (if none, don't get)
               labels_kitti,            # labels in data
               color_map_kitti,         # color for each label
               learning_map_kitti,      # mapping for training labels
               learning_map_inv_kitti,  # recover labels from xentropy
               sensor_kitti,            # sensor to use
               max_points_kitti,        # max points in each scan in entire dataset
               batch_size,        # batch size for train and val
               workers,           # threads to load data
               gt=True,           # get gt?
               shuffle_train=True,
               iswaymo = False)
    
    parser_waymo = Parser(root_waymo,              # directory for data
               train_sequences,   # sequences to train
               valid_sequences,   # sequences to validate.
               test_sequences,    # sequences to test (if none, don't get)
               labels_waymo,            # labels in data
               color_map_waymo,         # color for each label
               learning_map_waymo,      # mapping for training labels
               learning_map_inv_waymo,  # recover labels from xentropy
               sensor_waymo,            # sensor to use
               max_points_waymo,        # max points in each scan in entire dataset
               batch_size,        # batch size for train and val
               workers,           # threads to load data
               gt=True,           # get gt?
               shuffle_train=True,
               iswaymo = True)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    model.to(device)

    print(device, ' is used')
    
    
    trainloader = KittiWaymoTrainLoader(parser_kitti,parser_waymo)

    kvalidation_dataloader = parser_kitti.validloader

    wvalidation_dataloader = parser_waymo.validloader

    frequencies = kitti_normalized_frequencies
    frequencies_waymo = [i/sum(waymovalidfreqs) for i in waymovalidfreqs]

    inverse_frequencies = [1.0 / (f + epsilon_w) for f in frequencies]
    inverse_frequencies[0] = min(inverse_frequencies) / 50
    criterion_nll = nn.NLLLoss(weight=torch.tensor(inverse_frequencies).to(device))
    criterion_lovasz = Lovasz_softmax(ignore=0, from_logits=False)

    #For Waymo
    inverse_frequencies_waymo = [1.0 / (f + epsilon_w) for f in frequencies_waymo]
    inverse_frequencies_waymo[0] = min(inverse_frequencies_waymo) / 50
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
    
    num_classes_waymo = 23
    num_classes_kitti = 20
    best_mean_iou_waymo = 0.0
    metric_waymo = iouEval(num_classes_waymo,device,0)
    best_mean_iou_kitti = 0.0
    metric_kitti = iouEval(num_classes_kitti,device,0)

    # Training loop
    for epoch in range(max_epochs):
        model.train()
        running_loss = 0.0

        # Learning rate scheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        for i, (batchk, batchw) in tqdm(enumerate(trainloader), total=len(trainloader)):

            imagek, maskk = batchk
            imagew, maskw = batchw
            imagekw = torch.cat((imagek,imagew), dim=0)

            maskk = maskk.to(torch.long).to(device)
            maskw = maskw.to(torch.long).to(device)
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
            if DEBUG:
                if i>10:
                    break

        print(f"Epoch [{epoch+1}/{max_epochs}], Training Loss: {running_loss / len(trainloader):.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Validation phase
        model.eval()
        miou_total_kitti = 0.0
        miou_total_waymo = 0.0
        batch_results_kitti, batch_results_waymo = [], [] 

        metric_waymo.reset()
        metric_kitti.reset()

        with torch.no_grad():
            for i, (images, masks) in tqdm(enumerate(kvalidation_dataloader)):
                images, masks = images.to(torch.float32).to(device), masks.to(torch.long).to(device)

                # Forward pass and predictions
                outputs = model(images)[0]
                _, preds = torch.max(outputs, dim=1)
                metric_kitti.addBatch(preds,masks)
                if DEBUG:
                    if i>10:
                        break


            for i, (images, masks) in tqdm(enumerate(wvalidation_dataloader)):
                images, masks = images.to(torch.float32).to(device), masks.to(torch.long).to(device)

                # Forward pass and predictions
                outputs = model(images)[1]
                _, preds = torch.max(outputs, dim=1)
                metric_waymo.addBatch(preds,masks)
                if DEBUG:
                    if i>10:
                        break

        mean_iou_kitti = metric_kitti.getIoU()[0].item()
        accval_kitti = metric_kitti.getacc().item()

        mean_iou_waymo = metric_waymo.getIoU()[0].item()
        accval_waymo = metric_waymo.getacc().item()

        # Calculate and display mIoU metrics
        print('###################KITTI_RESULTS########################')
        print(f"Epoch [{epoch+1}/{max_epochs}], Kitti Validation mIoU: {mean_iou_kitti:.4f}")
        print(f"Epoch [{epoch+1}/{max_epochs}], Kitti Validation Acc.: {accval_kitti:.4f}")
        print('########################################################')

        # Calculate and display mIoU metrics
        print('###################WAYMO_RESULTS########################')
        print(f"Epoch [{epoch+1}/{max_epochs}], Waymo Validation mIoU: {mean_iou_waymo:.4f}")
        print(f"Epoch [{epoch+1}/{max_epochs}], Waymo Validation Acc.: {accval_waymo:.4f}")
        print('########################################################')

        if mean_iou_kitti>best_mean_iou_kitti:
            best_mean_iou_kitti = mean_iou_kitti
            print('The best results: ', metric_kitti.getIoU(), metric_kitti.getacc())

            qat_state_dict = model.state_dict()
            filtered_state_dict = {key: value for key, value in qat_state_dict.items() if key in original_state_dict}
            float_model.load_state_dict(filtered_state_dict)  # Load back weights
            torch.save(float_model.state_dict(), "float_best_kitti.pth")  # Save the original model
            torch.save(model.state_dict(), "qat_best_kitti.pth")
            print("Kitti model is saved.")
        
        if mean_iou_waymo>best_mean_iou_waymo:
            best_mean_iou_waymo = mean_iou_waymo
            print('The best results: ', metric_waymo.getIoU(), metric_waymo.getacc())

            qat_state_dict = model.state_dict()
            filtered_state_dict = {key: value for key, value in qat_state_dict.items() if key in original_state_dict}
            float_model.load_state_dict(filtered_state_dict)  # Load back weights
            torch.save(float_model.state_dict(), "float_best_waymo.pth")  # Save the original model
            torch.save(model.state_dict(), "qat_best_waymo.pth")
            print("waymo model is saved")

    torch.save(model.cpu().state_dict(), 'qt_last_model_state_dict.pth')


