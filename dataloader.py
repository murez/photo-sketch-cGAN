import os
import os.path
import numpy as np
from os import listdir
from os.path import join

import torch
from torch.autograd import Variable
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
    def __init__(self, photo_dir='training_data/photo/', sketch_dir='training_data/sketch/', loadSize = 256, fineSize = 256):
        super(CUFS, self).__init__()
        self.photo_list = [x for x in listdir(photo_dir) if is_image_file(x)]
        self.sketch_list = [x for x in listdir(sketch_dir) if is_image_file(x)]
        self.photo_sketch_dict = dict(zip(listdir(photo_dir), listdir(sketch_dir)))
        self.photo_dir = photo_dir
        self.sketch_dir = sketch_dir
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.photo_matrix = Variable(torch.FloatTensor())   # Will be initialized in train()
        self.sketch_matrix = Variable(torch.FloatTensor())
        self.hidden_photo_matrix = Variable(torch.FloatTensor())
        self.hidden_sketch_matrix = Variable(torch.FloatTensor())

    def __getitem__(self, index):
        photo_path = os.path.join(self.photo_dir, self.photo_list[index])
        sketch_path = os.path.join(self.sketch_dir, self.photo_sketch_dict[self.photo_list[index]])
        photo = default_loader(photo_path)
        sketch = default_loader(sketch_path)
        width, height = photo.size

        if (height != self.loadSize):
            photo = photo.resize((self.loadSize, self.loadSize), Image.BILINEAR)
            sketch = sketch.resize((self.loadSize, self.loadSize), Image.BILINEAR)

        if (self.loadSize != self.fineSize):
            x = math.floor((self.loadSize - self.fineSize)/2)
            y = math.floor((self.loadSize - self.fineSize)/2)
            photo = photo.crop((x, y, x+self.fineSize, y+self.fineSize))
            sketch = sketch.crop((x, y, x+self.fineSize, y+self.fineSize))

        # photo.show()
        photo = to_tensor(photo)
        sketch = to_tensor(sketch)
        item = {'photo': photo, 'sketch': sketch}
        return item

    def __len__(self):
        assert(len(self.photo_list) != len(self.sketch_list)), "More photos than sketches"
        return len(self.photo_list)

def get_loader(photo_dir, sketch_dir, batch_size, scale_size, num_workers=12, shuffle=True):
    dataset = CUFS(photo_dir, sketch_dir, scale_size, scale_size)
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader
