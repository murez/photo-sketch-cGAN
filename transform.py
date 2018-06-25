import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix
from sparsesvd import sparsesvd
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

def to_patch_matrices(CUFS_data, channel=0):
    photo_matrix = to_patch(CUFS_data[0]['photo'], channel=channel)
    sketch_matrix = to_patch(CUFS_data[0]['sketch'], channel=channel)
    for idx in range(len(CUFS_data)):
        if (idx == 0):
            continue
        photo = to_patch(CUFS_data[idx]['photo'], channel=channel) # photo and sketch tensor
        sketch = to_patch(CUFS_data[idx]['sketch'], channel=channel)
        photo_matrix = torch.cat((photo_matrix, photo), 1)
        sketch_matrix = torch.cat((sketch_matrix, sketch), 1)
    return photo_matrix, sketch_matrix

def to_patch(tensor, patch_size=13, r=2/3, dim=1, channel=0):
    '''
    Divides a torch.tensor into overlapping patches (r region overlapping),
    and stack them together.
    Input:
    tensor: two dimensional torch.tensor
    '''
    tensor = tensor[channel, :, :]
    # tensor = tensor.transpose(0,1)

    overlapping_size = math.sqrt(math.pow(patch_size,2) * (1-r))
    overlapping_size = (int)((patch_size - overlapping_size) / 2)
    increment = patch_size - overlapping_size
    output = Variable(torch.FloatTensor())
    y = 0
    while (y < tensor.size()[1] - patch_size):
        # Divide tensor into patches along x-dimension
        x = 0
        while (x < tensor.size()[0] - patch_size):
            output = torch.cat((output, tensor[y:y+patch_size, x:x+patch_size]), dim)
            x += increment
        output = torch.cat((output, tensor[y:y+patch_size, (tensor.size()[0]-patch_size):]), dim)
        y += increment

    x = 0
    while (x < tensor.size()[0] - patch_size):
        output = torch.cat((output, tensor[(tensor.size()[1]-patch_size):, x:x+patch_size]), dim)
        x += increment
    output = torch.cat((output, tensor[(tensor.size()[1]-patch_size):, (tensor.size()[0]-patch_size):]), dim)

    return output

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
                row_tensor = torch.cat((row_tensor, patch_matrix[0:increment, patch_count*patch_size:patch_count*patch_size+increment]), 1)
            else:
                row_tensor = torch.cat((row_tensor, patch_matrix[0:increment, (patch_count+1)*patch_size-remainder:(patch_count+1)*patch_size]), 1)
            patch_count += 1
        # end for
        image_tensor = torch.cat((image_tensor, row_tensor), 0)
    # end for

    # Last row
    row_tensor = Variable(torch.FloatTensor())
    for x in range(num_patches):
        if (x < num_patches - 1):
            row_tensor = torch.cat((row_tensor, patch_matrix[patch_size-remainder:patch_size, patch_count*patch_size:patch_count*patch_size+increment]), 1)
        else:
            row_tensor = torch.cat((row_tensor, patch_matrix[patch_size-remainder:patch_size, (patch_count+1)*patch_size-remainder:(patch_count+1)*patch_size]), 1)
        patch_count += 1
    # end for
    image_tensor = torch.cat((image_tensor, row_tensor), 0)

    return image_tensor

def calculate_weight(x_prime, x_o, x_h, empty=False):
    '''
    Input:
    x_prime: test photo patch matrix
    x_o: original training photo patch matrix
    x_h: hidden training photo patch matrix obtained from GAN
    Output: weight matrix of type torch.tensor
    '''
    x = torch.cat((x_prime, x_o, x_h), 1)
    x = x.detach().numpy()
    sx = scipy.sparse.csc_matrix(x)
    U, Sigma, V_T = sparsesvd(sx, 13)
    V = np.transpose(V_T)

    V_x_prime = V[0:x_prime.size(1), :]
    V_x_o = V[x_prime.size(1):(x_prime.size(1)+x_o.size(1)), :]
    print(V_x_prime.shape, V_x_o.shape)
    W_oh = np.dot(V_x_o, np.transpose(V_x_prime))
    # W_oh = sparse.csc_matrix(V_x_o).dot(sparse.csc_matrix(np.transpose(V_x_prime)))
    W_oh = torch.from_numpy(W_oh)
    print('W_oh size: ', W_oh.size())

    V_x_h = V[(x_prime.size(1)+x_o.size(1)):, :]
    W_ho = np.dot(V_x_h, np.transpose(V_x_prime))
    W_ho = torch.from_numpy(W_ho)
    print('W_ho size: ', W_ho.size())

    # return W_oh, W_ho
    return torch.cat((W_oh, W_ho), 0) # Concatenate along the row

def matrix_multiplication(x, y):
    '''
    Input: numpy array
    '''
    step = y.shape[1]
    size = x.shape[0]
    total_steps = (int)(size / step)

    mini_x = csr_matrix(x[0:step, :])
    y = csr_matrix(y)
    output = np.empty([size, step])
    output[0:step, :] = mini_x.dot(y).A
    for i in range(total_steps):
        if (i==0):
            continue
        else:
            mini_x = csr_matrix(x[i*step:(i+1)*step, :])
            output[i*step:(i+1)*step, :] = mini_x.dot(y).A
    return output
