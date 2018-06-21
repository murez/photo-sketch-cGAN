import numpy as np
import os
from os.path import join
from os import listdir

import torch
from torch.autograd import Variable
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
        photo = to_patch(CUFS_data[idx]['photo']) # photo and sketch tensor
        sketch = to_patch(CUFS_data[idx]['sketch'])
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

    # tensor = (tensor[0,:,:] + tensor[1,:,:] + tensor[2,:,:]).div(3)
    tensor = tensor.transpose(0,1)

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

'''Is this correct?'''
def inverse_to_patch(patch_matrix, patch_size=13, scale_size=256, r=2/3):
    '''
    Inverse transformation for to_patch()
    '''
    overlapping_size = math.sqrt(math.pow(patch_size,2)*(1-r))
    overlapping_size = (int)((patch_size-overlapping_size) / 2)
    increment = patch_size - overlapping_size
    num_patches = (int)((scale_size-overlapping_size)/increment) + 1
    remainder = scale_size - increment*(num_patches-1)
    patch_count = 0

    image_tensor = Variable(torch.FloatTensor())
    for y in range(num_patches - 1):
        row_tensor = Variable(torch.FloatTensor())
        for x in range(num_patches):
            if (x < num_patches - 1):
                row_tensor = torch.cat((row_tensor, patch_matrix[0:increment, patch_count*scale_size:patch_count*scale_size+increment]), 1)
            else:
                row_tensor = torch.cat((row_tensor, patch_matrix[0:increment, (patch_count+1)*scale_size-remainder:(patch_count+1)*scale_size]), 1)
            patch_count += 1
        # end for
        image_tensor = torch.cat((image_tensor, row_tensor), 0)
    # end for

    # Last row
    row_tensor = Variable(torch.FloatTensor())
    for x in range(num_patches):
        if (x < num_patches - 1):
            row_tensor = torch.cat((row_tensor, patch_matrix[scale_size-remainder:scale_size, patch_count*scale_size:patch_count*scale_size+increment]), 1)
        else:
            row_tensor = torch.cat((row_tensor, patch_matrix[scale_size-remainder:scale_size, (patch_count+1)*scale_size-remainder:(patch_count+1)*scale_size]), 1)
        patch_count += 1
    # end for
    image_tensor = torch.cat((image_tensor, row_tensor), 0)

    return image_tensor

def calculate_weight(x_prime, x_o, x_h):
    '''
    Input:
    x_prime: test photo patch matrix
    x_o: original trainig photo patch matrix
    x_h: hidden training photo patch matrix obtained from GAN
    inputs can't be empty!!!
    Output: weight matrix of type torch.tensor
    '''
    x = torch.cat((x_prime, x_o, x_h), 1)
    x = x.numpy()
    U, Sigma, V_x_T = np.linalg.svd(x, full_matrices=True)

    V_x_prime = np.transpose(V_x_T)[:, 0:x_prime.size(1)]
    V_x_o = np.transpose(V_x_T)[:, x_prime.size(1): x_prime.size(1)+x_o.size(1)]
    V_x_h = np.transpose(V_x_T)[:, x_prime.size(1)+x_o.size(1):]
    W_oh = np.dot(np.transpose(V_x_o_T), V_x_prime_T)
    W_ho = np.dot(np.transpose(V_x_h_T), V_x_prime_T)
    W_oh = torch.from_numpy(W_oh)
    W_ho = torch.from_numpy(W_ho)
    return torch.cat((W_oh, W_ho), 0) # Concatenate along the row
