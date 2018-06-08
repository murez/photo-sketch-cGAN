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
        self.act_type = args.tanh
        self.disc = disc

        self.l0 = nn.Linear(self.h, 8*8*self.num_channel)   #Fully Connected
        self.conv1 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1) #padding=1
        self.conv1_bn = nn.BatchNorm2d(self.num_channel)    #Batch Normalization
        self.conv2 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.conv2_bn = nn.BatchNorm2d(self.num_channel)
        self.up1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv3 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.conv3_bn = nn.BatchNorm2d(self.num_channel)
        self.conv4 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.conv4_bn = nn.BatchNorm2d(self.num_channel)
        self.up2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv5 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.conv5_bn = nn.BatchNorm2d(self.num_channel)
        self.conv6 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.conv6_bn = nn.BatchNorm2d(self.num_channel)
        self.up3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv7 = nn.Conv2d(self.num_channel, 3, 3, 1, 1)


    def forward(self, input):
        '''
        Use elu as activation function.
        '''
        x = self.l0(input)
        x = x.view(self.b_size, self.num_channel, 8, 8) #Reshape
        x = F.elu(self.conv1_bn(self.conv1(x)), True)
        x = F.elu(self.conv2_bn(self.conv2(x)), True)
        x = self.up1(x)

        x = F.elu(self.conv3_bn(self.conv3(x)), True)
        x = F.elu(self.conv4_bn(self.conv4(x)), True)
        x = self.up2(x)

        x = F.elu(self.conv5_bn(self.conv5(x)), True)
        x = F.elu(self.conv6_bn(self.conv6(x)), True)
        x = F.tanh(self.conv7(x), True)
        return x

class Discriminator(nn.Module):
    '''
    Input image: 32*32*3
    '''
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.num_channel = args.nc
        self.b_size = args.b_size
        self.h = args.h
        self.scale_size = args.scale_size

        self.conv0 = nn.Conv2d(3, self.num_channel, 3, 1, 1)
        self.conv1 = nn.Conv2d(self.num_channel, self.num_channel, 3, 1, 1)
        self.conv2 = nn.Conv2d(self.num_channel, 2*self.num_channel, 3, 1, 1)
        self.pool1 = nn.AvgPool2d((2, 2))   #Subsampling
        self.conv3 = nn.Conv2d(2*self.num_channel, 2*self.num_channel, 3, 1, 1)
        self.conv4 = nn.Conv2d(2*self.num_channel, 3*self.num_channel, 3, 1, 1)
        self.pool2 = nn.AvgPool2d((2,2))
        self.conv5 = nn.Conv2d(3*self.num_channel, 3*self.num_channel, 3, 1, 1)
        self.conv6 = nn.Conv2d(3*self.num_channel, 3*self.num_channel, 3, 1, 1)
        self.fc = nn.Linear(8*8*3*self.num_channel, self.h)

    def forward(self, input):
        x = F.elu(self.conv0(input), True)
        x = F.elu(self.conv1(input), True)
        x = F.elu(self.conv2(input), True)
        x = self.pool1(x)

        x = F.elu(self.conv3(input), True)
        x = F.elu(self.conv4(input), True)
        x = self.pool2(x)

        x = F.elu(self.conv5(input), True)
        x = F.elu(self.conv6(input), True)
        x = x.view(self.b_size, 8*8*3*self.num_channel)
        x = F.elu(self.fc(x), True)
        return x

'''
class GAN(nn.Module):
    def __init__(self, nc):
        super(GAN, self).__init__()
        self.generator = Generator(nc, True)
        self.discriminator = Discriminator(nc)
    def forward(self, input):
        return self.generator(self.discriminator(input))

class Loss(nn.Module):
    def __init__(self, size_avg = True):
        super(Loss, self).__init__()
        self.size_avg = size.avg
'''
