import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import argparse
import cv2
import glob
import os
import numpy as np
import sys
from PIL import Image
import pytorch_nndct.nn as nndct_nn
from pytorch_nndct.nn.modules import functional

class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.ReLU6()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3), padding=1)
        self.act2 = nn.ReLU6()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act3 = nn.ReLU6()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.skip_add = functional.Add()


    def forward(self, x):

        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.bn1(resA)
        resA1 = self.act2(resA)

        resA = self.conv3(resA1)
        resA = self.bn2(resA)
        resA2 = self.act3(resA)

        output = self.skip_add(shortcut, resA2)
        return output


class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.ReLU6()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3,3), padding=1)
        self.act2 = nn.ReLU6()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3,3),dilation=2, padding=2)
        self.act3 = nn.ReLU6()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.ReLU6()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.ReLU6()
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

        self.skip_add = functional.Add()
        self.cat = functional.Cat()

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.bn1(resA)
        resA1 = self.act2(resA)

        resA = self.conv3(resA1)
        resA = self.bn2(resA)
        resA2 = self.act3(resA)

        resA = self.conv4(resA2)
        resA = self.bn3(resA)
        resA3 = self.act4(resA)

        concat = self.cat((resA1,resA2,resA3),dim=1)
        resA = self.conv5(concat)
        resA = self.bn4(resA)
        resA = self.act5(resA)
        resA = self.skip_add(shortcut,resA)


        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)

            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB


class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(in_filters//4 + 2*out_filters, out_filters, (3,3), padding=1)
        self.act1 = nn.ReLU6()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act2 = nn.ReLU6()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2,2), dilation=2,padding=1)
        self.act3 = nn.ReLU6()
        self.bn3 = nn.BatchNorm2d(out_filters)


        self.conv4 = nn.Conv2d(out_filters*3,out_filters,kernel_size=(1,1))
        self.act4 = nn.ReLU6()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.convup = nn.Conv2d(in_filters, in_filters//4, kernel_size=(1,1))
        self.bnup = nn.BatchNorm2d(in_filters//4)
        self.actup = nn.ReLU6()
        self.cat1 = functional.Cat()
        self.cat2 = functional.Cat()

    def forward(self, x, skip):
        #upA = nn.PixelShuffle(2)(x)
        upA = self.up(x)
        upA = self.convup(upA)
        upA = self.bnup(upA)
        upA = self.actup(upA)

        if self.drop_out:
            upA = self.dropout1(upA)

        upB = self.cat1((upA,skip),dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.bn1(upE)
        upE1 = self.act1(upE)

        upE = self.conv2(upE1)
        upE = self.bn2(upE)
        upE2 = self.act2(upE)

        upE = self.conv3(upE2)
        upE = self.bn3(upE)
        upE3 = self.act3(upE)

        concat = self.cat2((upE1,upE2,upE3),dim=1)
        upE = self.conv4(concat)
        upE = self.bn4(upE)
        upE = self.act4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE

# Define the Encoder part
class SalsaEncoder(nn.Module):
    def __init__(self, inchannels=5):
        super(SalsaEncoder, self).__init__()
        self.inchannels = inchannels
        self.downCntx = ResContextBlock(inchannels, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)

        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)

    def forward(self, x):
        downCntx = self.downCntx(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down5c = self.resBlock5(down3c)

        return down5c, [down3b, down2b, down1b, down0b]  # Feature maps to pass to the decoder


# Define the Decoder part
class SalsaDecoder(nn.Module):
    def __init__(self, nclasses=20):
        super(SalsaDecoder, self).__init__()
        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2)
        self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out=False)

        self.logits = nn.Conv2d(32, nclasses, kernel_size=(1, 1))

    def forward(self, x, encoder_features):
        down3b, down2b, down1b, down0b = encoder_features
        
        up4e = self.upBlock1(x, down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)
        logits = self.logits(up1e)

        return logits


class Salsa(nn.Module):
    def __init__(self, nclasses=[20,23], inchannels=5):
        super(Salsa, self).__init__()
        self.encoder = SalsaEncoder(inchannels)
        self.decoder1 = SalsaDecoder(nclasses[0])
        self.decoder2 = SalsaDecoder(nclasses[1])

    def forward(self, x):
        encoder_output, encoder_features = self.encoder(x)
        logits1 = self.decoder1(encoder_output, encoder_features)
        logits2 = self.decoder2(encoder_output, encoder_features)
        return logits1, logits2


class HydraSalsa(nn.Module):
    def __init__(self, nclasses=[20,23], inchannels=5):
        super(HydraSalsa, self).__init__()
        self.salsa = Salsa(nclasses,inchannels)
        self.quant_stub = nndct_nn.QuantStub()
        self.dequant_stub = nndct_nn.DeQuantStub()
    def forward(self, x):
        x = self.quant_stub(x)
        logits1,logits2 = self.salsa(x)
        logits1 = self.dequant_stub(logits1)
        logits2 = self.dequant_stub(logits2)
        probs1,probs2 = F.softmax(logits1, dim=1), F.softmax(logits2, dim=1)
        return probs1, probs2


class SalsaK(nn.Module):
    def __init__(self, nclasses=20, inchannels=5):
        super(SalsaK, self).__init__()
        self.encoder = SalsaEncoder(inchannels)
        self.decoder1 = SalsaDecoder(nclasses)

    def forward(self, x):
        encoder_output, encoder_features = self.encoder(x)
        logits1 = self.decoder1(encoder_output, encoder_features)
        return logits1

class SalsaNext(nn.Module):
    def __init__(self, nclasses=20, inchannels=5):
        super(SalsaNext, self).__init__()
        self.salsa = SalsaK(nclasses,inchannels)
        self.quant_stub = nndct_nn.QuantStub()
        self.dequant_stub = nndct_nn.DeQuantStub()

    def forward(self, x):
        x = self.quant_stub(x)
        logits1 = self.salsa(x)
        logits1 = self.dequant_stub(logits1)
        return logits1
