import numpy as np
import os
from os.path import join
from os import listdir

import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image

import random
import math

from dataloader import *

def to_patch_matrices(CUFS_data):
    photo_matrix = to_patch(CUFS_data[0]['photo'])
    sketch_matrix = to_patch(CUFS_data[0]['sketch'])
    for idx in range(len(CUFS_data)):
        if (idx == 0):
            continue
        photo, sketch = CUFS_data[idx] # photo and sketch tensor
        photo_matrix = torch.cat((photo_matrix, photo), 1)
        sketch_matrix = torch.cat((sketch_matrix, sketch), 1)
    return photo_matrix, sketch_matrix

def to_patch(tensor, patch_size=13, r=2/3, dim=1):
    '''
    Divides a torch.tensor into overlapping patches (r region overlapping),
    and stack them together.
    Input:
    tensor: two dimensional torch.tensor
    '''
    '''
    tensor = (tensor[0,:,:] + tensor[1,:,:] + tensor[2,:,:]).div(3)
    tensor = tensor.transpose(0,1)
    '''
    overlapping_size = math.sqrt(math.pow(patch_size,2) * (1-r))
    overlapping_size = (int)((patch_size - overlapping_size) / 2)
    increment = patch_size - overlapping_size
    output = tensor[0:patch_size, 0:patch_size]
    y = 0
    while (y < tensor.size()[1] - patch_size):
        # Divide tensor into patches along x-dimension
        x = 0
        while (x < tensor.size()[0] - patch_size):
            output = torch.cat((output, tensor[x:x+patch_size, y:y+patch_size]), dim)
            x += increment
        output = torch.cat((output, tensor[(tensor.size()[0]-patch_size):, y:y+patch_size]), dim)
        y += increment

    x = 0
    while (x < tensor.size()[0] - patch_size):
        output = torch.cat((output, tensor[x:x+patch_size, (tensor.size()[1]-patch_size):]), dim)
        x += increment
    output = torch.cat((output, tensor[(tensor.size()[0]-patch_size):, (tensor.size()[1]-patch_size):]), dim)

    output = output[:,patch_size:]
    return output

def calculate_weight(x_prime, x_o, x_h):
    '''
    Input:
    x_prime: test photo patch matrix
    x_o: original trainig photo patch matrix
    x_h: hidden training photo patch matrix obtained from GAN
    Output: weight matrix of type torch.tensor
    '''
    x_prime = x_prime.numpy()
    x_o = x_o.numpy()
    x_h = x_h.numpy()
    U, Sigma, V_x_prime_T = np.linalg.svd(x_prime, full_matrices = True)
    U, Sigma, V_x_o_T = np.linalg.svd(x_o, full_matrices = True)
    U, Sigma, V_x_h_T = np.linalg.svd(x_h, full_matrices = True)
    W_oh = np.dot(np.transpose(V_x_o_T), V_x_prime_T)
    W_ho = np.dot(np.transpose(V_x_h_T), V_x_prime_T)
    W_oh = torch.from_numpy(W_oh)
    W_ho = torch.from_numpy(W_ho)
    return torch.cat((W_oh, W_ho), 0) # Concatenate along the row
