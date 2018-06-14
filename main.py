import argparse
import random
import os, time, sys
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.utils as vutils
from torchvision import transforms
import numpy as np
from collections import deque

from dataloader import *
from models import *
from transform import *

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true')      # gpu device
parser.add_argument('--b_size', default=16, type=int)   # batch size
parser.add_argument('--h', default=64, type=int)        # latent size
parser.add_argument('--nc', default=64, type=int)       # number of channel
parser.add_argument('--epochs', default=100, type=int)  # epochs
parser.add_argument('--lr',default=0.0001, type=float)  # learning rate
# parser.add_argument('--lr_update_step', default=3000, type=float)
parser.add_argument('--lr_update_type', default=1, type=int)
parser.add_argument('--scale_size', default=64, type=int)
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--beta_lo', default=0.5, type=float)
parser.add_argument('--beta_hi',default=0.999, type=float)
parser.add_argument('--sample_dir', default='sample/', type=str)
parser.add_argument('--step_size', default=30, type=int)    # learning rate optimizer step size
parser.add_argument('--gamma', default=0.1, type=float)      # learning rate optimizer decay parameter
parser.add_argument('--base_path', default="training_data/", type=str)
parser.add_argument('--photo_dir', default="photo/", type=str)
parser.add_argument('--sketch_dir', default="sketch/", type=str)


args = parser.parse_args()


def onehot_encoding(x, num_classes=10):
    if isinstance(x, int):
        y = torch.zeros(1, num_classes).long()
        y[0][x] = 1
    else:
        x = x.cpu()
        y = torch.LongTensor(x.size(0), num_classes)
        y.zero_()
        y.scatter_(1,x,1)   # dim, index, src value
    return y

class GAN():
    def __init__(self):
        # self.global_step = args.load_step
        self.prepare_paths()
        self.dataset = CUFS(self.photo_path, self.sketch_path, args.scale_size, args.scale_size)
        self.data_loader = torch.utils.data.DataLoader(dataset=self.dataset, batch_size=args.b_size, shuffle=True, num_workers=args.num_workers)
        self.build_model()
        self.z = Variable(torch.FloatTensor(args.b_size, args.h, args.scale_size, args.scale_size))

        self.criterion = nn.BCELoss()
        if args.cuda:
            self.set_cuda()

    def set_cuda(self):
        self.discriminator.cuda()
        self.generator.cuda()
        self.z = self.z.cuda()
        self.criterion.cuda()

    def prepare_paths(self):
        self.photo_path = os.path.join(args.base_path, args.photo_dir)
        self.sketch_path = os.path.join(args.base_path, args.sketch_dir)

    def build_model(self):
        self.discriminator = Discriminator(args)
        self.generator = Generator(args)
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), betas=(args.beta_lo,args.beta_hi), lr=args.lr)
        self.g_optimizer = torch.optim.Adam(self.g.parameters(), betas=(args.beta_lo,args.beta_hi), lr=args.lr)
        self.d_scheduler = torch.optim.lr_scheduler.StepLR(self.d_optimizer, step_size = args.step_size, gamma = args.gamma)
        self.g_scheduler = torch.optim.lr_scheduler.StepLR(self.g_optimizer, step_size = args.step_size, gamma = args.gamma)

    def reset_gradient(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def train(self):
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(args.sample_dir):
            os.mkdir(args.sample_dir)

        train_dict = {}
        train_dict['D_losses'] = []
        train_dict['G_losses'] = []
        train_dict['time_per_epoch'] = []
        train_dict['total_time'] = []

        print('Start training!')
        start_time = time.time()

        for epoch in range(args.epochs):
            D_losses = []
            G_losses = []
            # Learning rate decay
            self.d_scheduler.step()
            self.g_scheduler.step()

            real_labels = torch.ones(args.b_size, args.h).to(device)
            fake_labels = torch.zeros(args.b_size, args.h).to(device)
            epoch_start_time = time.time()
            num_iter = 0

            for idx, (photo, sketch) in enumerate(self.data_loader):
                step += 1
                # ================================================== #
                #               Train the Discriminator              #
                # ================================================== #
                # Compute BCE_Loss using real images
                # y is input, x is label
                x = photo.to(device)
                y = sketch.to(device)
                # y = sketch.view(args.b_size, 1)
                # y = onehot_encoding(y, num_classes=10).to(device)
                output_x = self.discriminator(y,x)
                d_loss_real = self.criterion(output_x, real_labels)

                # Compute BCE_Loss using fake images
                self.z.data.uniform_(-1,1).to(device)   # z.size = args.b_size, args.h
                fake_image = self.generator(self.z, x)
                output_z = self.discriminator(fake_image, x)
                d_loss_fake = self.criterion(output_z, fake_labels)
                d_loss = d_loss_real + d_loss_fake
                D_losses.append(d_loss.data[0])

                # Backpropagation and optimization
                self.reset_gradient()
                d_loss.backward()
                self.d_optimizer.step()

                # ================================================= #
                #                Train the Generator                #
                # ================================================= #
                # Train G to maximize log(D(G(z))) instead of minimizing log(1-D(G(z)))
                g_loss = self.criterion(output_z, real_labels)
                G_losses.append(g_loss.data[0])

                # Backpropagation and optimization
                self.reset_gradient()
                g_loss.backward()
                self.g_optimizer.step()
            # end for

            epoch_end_time = time.time()
            time_per_epoch = epoch_end_time - epoch_start_time

            # Print epoch result on the terminal
            print('Epoch: [%d/%d] - time_per_epoch: %.2f, d_loss: %.3f, g_loss: %.3f' % (epoch + 1), args.epochs, time_per_epoch,
                                                                                    torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_losses)))
            # Save images
            # Save real images
            if (epoch+1) == 1:
                path = os.path.join(args.sample_dir, 'real_sketch.jpg')
                vutils.save_image(image.mul(255).byte(),path)

             # Save fake images
             epoch_save = 10   # After 10 epochs, starting saving fake images(sketches)
             if (epoch+1) > epoch_save:
                 path = os.path.join(args.sketch_path, 'fake_sketch-{}.jpg'.format(epoch+1))
                 # Calculate weight matrix to obtain corresponding photo, and add it to photo directory

                 vutils.save_image(fake_image.mul(255).byte(), path)

             train_dict['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
             train_dict['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))
             train_dict['time_per_epoch'].append(time_per_epoch)
        # end for

        end_time = time.time()
        total_time = end_time - start_time
        train_dict['total_time'].append(total_time)

        print('----------------------------------------------------------------')
        print('Average time per epoch: %.2f, total %d epochs time: %.2f' % (torch.mean(torch.FloatTensor(train_dict['time_per_epoch'])), args.epochs, total_time))
        print('Finish training!')
        print('Saving training results...')

        # Save the model checkpoints
        torch.save({'epoch': args.epochs, 'model': 'discriminator', 'model_state_dict': self.discriminator.state_dict(),
                                'optimizer': self.d_optimizer.state_dict()}, 'discriminator_parameter_checkpoint.pth.tar')
        torch.save({'epoch': args.epochs, 'model': 'generator', 'model_state_dict': self.generator.state_dict(),
                                    'optimizer': self.g_optimizer.state_dict()}, 'generator_parameter_checkpoint.pth.tar')
        print('Training results saved!')

# main function
if __name__ == "__main__":
    gan = GAN()
    gan.train()
