import os
import argparse
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets

class Generator(nn.Module):
    '''
    Use BEGAN model's decoder.
    Use an auto-encoder with both a deep encoder and decoder.
    '''
    def __init__(self, args, disc = False):
        super(Generator, self).__init__()
        self.num_channel = args.nc
        self.b_size = args.b_size
        self.h = args.h
        self.scale_size = args.scale_size
        self.disc = disc

        ''' Use transpose convolution '''
        self.deconv0_1 = nn.ConvTranspose2d(self.h, 4*self.num_channel, 3, 1, 1)
        self.deconv0_1_bn = nn.BatchNorm2d(4*self.num_channel)
        self.deconv0_2 = nn.Conv2d(3, 4*self.num_channel, 3, 1, 1)
        self.deconv0_2_bn = nn.BatchNorm2d(4*self.num_channel)
        self.deconv2 = nn.ConvTranspose2d(8*self.num_channel, 4*self.num_channel, 3, 1, 1)
        self.deconv2_bn = nn.BatchNorm2d(4*self.num_channel)
        self.deconv3 = nn.ConvTranspose2d(4*self.num_channel, 2*self.num_channel, 3, 1, 1)
        self.deconv3_bn = nn.BatchNorm2d(2*self.num_channel)
        self.deconv4 = nn.ConvTranspose2d(2*self.num_channel, self.num_channel, 3, 1, 1)
        self.deconv4_bn = nn.BatchNorm2d(self.num_channel)
        self.deconv5 = nn.ConvTranspose2d(self.num_channel, 3, 3, 1, 1)
        
    def forward(self, input, label):
        x = F.elu(self.deconv0_1_bn(self.deconv0_1(input)), True)
        y = F.elu(self.deconv0_2_bn(self.deconv0_2(label)), True)
        x = torch.cat((x,y), 1)
        x = F.elu(self.deconv2_bn(self.deconv2(x)), True)
        x = F.elu(self.deconv3_bn(self.deconv3(x)), True)
        x = F.elu(self.deconv4_bn(self.deconv4(x)), True)
        x = F.sigmoid(self.deconv5(x))
        return x

class Discriminator(nn.Module):
    '''
    Input image: b_size*3*256*256
    '''
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.num_channel = args.nc
        self.b_size = args.b_size
        self.h = args.h     # h = d
        self.scale_size = args.scale_size

        # Conditional GAN
        self.conv0_1 = nn.Conv2d(3, (int)(self.num_channel/2), 3, 1, 1)
        self.conv0_2 = nn.Conv2d(3, (int)(self.num_channel/2), 3, 1, 1)
        self.conv1 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.conv1_bn = nn.BatchNorm2d(self.num_channel)
        self.conv2 = nn.Conv2d(self.num_channel, 2*self.num_channel, 3, 1, 1)
        self.conv2_bn = nn.BatchNorm2d(2*self.num_channel)
        self.pool1 = nn.AvgPool2d((2, 2))   #Subsampling
        self.conv3 = nn.Conv2d(2*self.num_channel, 2*self.num_channel, 3, 1, 1)
        self.conv3_bn = nn.BatchNorm2d(2*self.num_channel)
        self.conv4 = nn.Conv2d(2*self.num_channel, 3*self.num_channel, 3, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(3*self.num_channel)
        self.pool2 = nn.AvgPool2d((2,2))
        self.conv5 = nn.Conv2d(3*self.num_channel, 3*self.num_channel, 3, 1, 1)
        self.conv5_bn = nn.BatchNorm2d(3*self.num_channel)
        self.conv6 = nn.Conv2d(3*self.num_channel, 3*self.num_channel, 3, 1, 1)
        self.conv6_bn = nn.BatchNorm2d(3*self.num_channel)
        self.fc = nn.Linear((self.scale_size/4)*(self.scale_size/4)*3*self.num_channel, self.h)
        # Need more layers?

    def forward(self, input, label):
        # input, label must be size of (b_size, C, H, W)
        x = F.elu(self.conv0_1(input), True)
        y = F.elu(self.conv0_2(label), True)
        x = torch.cat((x,y), 1)     # Discriminator conditions on label

        x = F.elu(self.conv1_bn(self.conv1(x)), True)
        x = F.elu(self.conv2_bn(self.conv2(x)), True)
        x = self.pool1(x)

        x = F.elu(self.conv3_bn(self.conv3(x)), True)
        x = F.elu(self.conv4_bn(self.conv4(x)), True)
        x = self.pool2(x)

        x = F.elu(self.conv5_bn(self.conv5(x)), True)
        x = F.elu(self.conv6_bn(self.conv6(x)), True)
        x = x.view(self.b_size, (self.scale_size/4)*(self.scale_size/4)*3*self.num_channel)
        x = F.sigmoid(self.fc(x))
        return x    # Size of (b_size, h), value should be between 0 and 1
