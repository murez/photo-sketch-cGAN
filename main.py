import argparse
import random
import os
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

parser = argparse.ArgumentParser()
parser.add_argument('--cuda', action='store_true')      # gpu device
parser.add_argument('--b_size', default=16, type=int)   # batch size
parser.add_argument('--h', default=64, type=int)        # latent size
parser.add_argument('--nc', default=64, type=int)
parser.add_argument('--epochs', default=100, type=int)  # epochs
parser.add_argument('--lr',default=0.0001, type=float)  # learning rate
# parser.add_argument('--lr_update_step', default=3000, type=float)
parser.add_argument('--lr_update_type', default=1, type=int)
parser.add_argument('--scale_size', default=64, type=int)
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--beta_lo', default=0.5, type=float)
parser.add_argument('--beta_hi',default=0.999, type=float)
parser.add_argument('--sample_dir', default='sample', type=str)


args = parser.parse_args()

class GAN():
    def __init__(self):
        # self.global_step = args.load_step
        # self.prepare_paths()
        self.data_loader = get_loader(self.data_path, args.b_size, args.scale_size, args.num_workers)
        self.build_model()
        self.z = Variable(torch.FloatTensor(args.b_size, args.h))

        self.criterion = nn.BCELoss()
        if args.cuda:
            self.set_cuda()

    def set_cuda(self):
        self.discriminator.cuda()
        self.generator.cuda()
        self.z = self.z.cuda()
        self.criterion.cuda()

    def build_model(self):
        self.discriminator = Discriminator(args)
        self.generator = Generator(args)
        self.d_optimizer = torch.optim.Adam(self.d.parameters(), betas=(args.beta_lo,args.beta_hi), lr=args.lr)
        self.g_optimizer = torch.optim.Adam(self.g.parameters(), betas=(args.beta_lo,args.beta_hi), lr=args.lr)

    def reset_gradient(self):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()

    def train(self):
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if not os.path.exists(args.sample_dir):
            os.mkdir(args.sample_dir)
        # g_optimizer = torch.optim.Adam(self.g.parameters(), betas = (0.5, 0.999), lr = args.lr)
        # d_optimizer = torch.optim.Adam(self.d.parameters(), betas = (0.5, 0.999), lr = args.lr)

        for epoch in range(args.epochs):
            for idx, (images, labels) in enumerate(self.data_loader):
                step += 1
                real_labels = torch.ones(args.b_size, 1).to(device)
                fake_labels = torch.zeros(args.b_size, 1).to(device)

                # ================================================== #
                #               Train the Discriminator              #
                # ================================================== #
                # Compute BCE_Loss using real images
                output_x = self.discriminator.forward(image)
                d_loss_real = self.criterion(output_x, real_labels)

                # Compute BCE_Loss using fake images
                self.z.data.uniform_(-1,1).to(device)
                fake_image = self.generator.forward(self.z)
                output_z = self.discriminator.forward(fake_image)
                d_loss_fake = self.criterion(output_z, fake_labels)

                # Backpropagation and optimization
                d_loss = d_loss_real + d_loss_fake
                self.reset_gradient()
                d_loss.backward()
                self.d_optimizer.step()

                # ================================================= #
                #                Train the Generator                #
                # ================================================= #
                # Train G to maximize log(D(G(z))) instead of minimizing log(1-D(G(z)))
                g_loss = self.criterion(output_z, real_labels)

                # Backpropagation and optimization
                self.reset_gradient()
                g_loss.backward()
                self.g_optimizer.step()

                if step % 1000 == 0:
                    print('Epoch: {}/{}, Step: {}, D Loss: {}, G Loss: {}'.format(epoch, arg.epochs, step, d_loss.data[0], g_loss.data[0]))

            # Save real images
            if (epoch+1) == 1:
                path = os.path.join(sample_dir, 'real_image.jpg')
                vutils.save_image(image.mul(255).byte(),path)

             # Save fake images
             path = os.path.join(args.sample_dir, 'fake_image-{}.jpg'.format(epoch+1))
             vutils.save_image(fake_image.mul(255).byte(), path)

        # Save the model checkpoints
        torch.save(self.discriminator.state_dict(), 'D.ckpt')
        torch.save(self.generator.state_dict(), 'G.ckpt')

# main function
if __name__ == "__main__":
    gan = GAN()
    gan.train()
