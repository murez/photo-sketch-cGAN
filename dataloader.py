import os
import os.path
import numpy as np
from os import listdir
from os.path import join

import torch
from torchvision import transforms
import torch.utils.data as data
from PIL import Image

import random
import math

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])

def default_loader(path):
    return Image.open(path).convert('RGB')

def to_tensor(picture):
    ''' Converts a PIL.Image or numpy.ndarray (H * W * C) in the range
    [0, 255] to a torch.FloatTensor of shape (C * H * W) in the range [0.0, 1.0]
    '''
    if isinstance(picture, np.ndarray):
        img = torch.from_numpy(np.transpose(picture, (2, 0, 1)))
        return img.float().div(225)
    if picture.mode == 'I':
        img = torch.from_numpy(np.array(picture, np.int32, copy=True))
    elif picture.mode == 'I;16':
        img = torch.from_numpy(np.array(picture, np.int16, copy=True))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(picture.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if picture.mode == 'YCbCr':
        num_channel = 3
    elif picture.mode == 'I;16':
        num_channel = 1
    else:
        num_channel = len(picture.mode)
    img = img.view(picture.size[1], picture.size[0], num_channel)
    img = img.transpose(0,1).transpose(0,2).contiguous()

    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img

class CUFS(data.Dataset):
    def __init__(self, dataPath='training_data/photo/', loadSize = 64, fineSize = 64, flip = 1):
        super(CUFS, self).__init__()
        self.data_list = [x for x in listdir(dataPath) if is_image_file(x)]
        self.dataPath = dataPath
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.flip = flip

    def __getitem__(self, index):
        path = os.path.join(self.dataPath, self.data_list[index])
        img = default_loader(path)
        width, height = img.size

        if (height != self.loadSize):
            img = img.resize((self.loadSize, self.loadSize), Image.BILINEAR)

        if (self.loadSize != self.fineSize):
            x = math.floor((self.loadSize - self.fineSize)/2)
            y = math.floor((self.loadSize - self.fineSize)/2)
            img = img.crop((x, y, x+self.fineSize, y+self.fineSize))

        #img.show()
        img = to_tensor(img)
        #img = img.mul_(2).add_(-1)

        return img

    def __len__(self):
        return len(self.data_list)

def get_loader(dir, batch_size, scale_size, num_workers=12, shuffle=True):
    dataset = CUFS(dir, scale_size, scale_size, 1)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader
