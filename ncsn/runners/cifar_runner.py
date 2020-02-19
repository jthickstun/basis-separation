import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from ncsn.models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader

__all__ = ['CifarRunner']

class CifarRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config

    def test(self):
        # Load the score network
        states = torch.load(os.path.join(self.args.log, 'checkpoint.pth'), map_location=self.config.device)
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet)
        scorenet.load_state_dict(states[0])
        scorenet.eval()

        # Grab some samples from the test set
        dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar'), train=False, transform=transforms.ToTensor(), download=True)
        dataloader = iter(DataLoader(dataset, batch_size=50, shuffle=False))
        x0,y0 = next(dataloader)
        x1,y1 = next(dataloader)

        self.write_images(x0, 'xgt.png')
        self.write_images(x1, 'ygt.png')

        mixed = (x0 + x1).cuda()

        self.write_images(mixed.cpu()/2., 'mixed.png')

        x0 = nn.Parameter(torch.Tensor(50,3,32,32).uniform_()).cuda()
        x1 = nn.Parameter(torch.Tensor(50,3,32,32).uniform_()).cuda()

        recon = (x0 + x1 - mixed)**2

        step_lr=0.00003

        # Noise amounts
        sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497,
                           0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01])
        n_steps_each = 100

        for idx, sigma in enumerate(sigmas):
            lambda_recon = 1./sigma**2
            labels = torch.ones(1, device=x0.device) * idx
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            
            print('sigma = {}'.format(sigma))
            for step in range(n_steps_each):
                recon = ((x0 + x1 - mixed)**2).view(-1,3*32*32).sum(1).mean()

                noise_x = torch.randn_like(x0) * np.sqrt(step_size * 2)
                noise_y = torch.randn_like(x1) * np.sqrt(step_size * 2)

                grad_x0 = scorenet(x0, labels).detach()
                grad_x1 = scorenet(x1, labels).detach()

                norm0 = np.linalg.norm(grad_x0.view(-1,3*32*32).cpu().numpy(),axis=1).mean()
                norm1 = np.linalg.norm(grad_x1.view(-1,3*32*32).cpu().numpy(),axis=1).mean()

                x0 += step_size * (grad_x0 - lambda_recon * (x0 + x1 - mixed)) + noise_x
                x1 += step_size * (grad_x1 - lambda_recon * (x0 + x1 - mixed)) + noise_y

            print(' recon: {}, |norm1|: {}, |norm2|: {}'.format(recon,norm0,norm1))

        # Write x0 and x1
        self.write_images(x0.detach().cpu(), 'x.png')
        self.write_images(x1.detach().cpu(), 'y.png')

    def write_images(self, x,name,n=7):
        x = x.numpy().transpose(0, 2, 3, 1)
        d = x.shape[1]
        panel = np.zeros([n*d,n*d,3],dtype=np.uint8)
        for i in range(n):
            for j in range(n):
                panel[i*d:(i+1)*d,j*d:(j+1)*d,:] = (256*(x[i*n+j])).clip(0,255).astype(np.uint8)[:,:,::-1]

        cv2.imwrite(os.path.join(self.args.image_folder, 'ncsn_cifar10_' + name), panel)
