import os

import cv2
import numpy as np
import torch
import torch.nn as nn

from ncsn.models.cond_refinenet_dilated import CondRefineNetDilated
from torchvision.datasets import MNIST, CIFAR10
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader

__all__ = ['CifarColorRunner']

BATCH_SIZE = 100
GRID_SIZE = 10

class CifarColorRunner():
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

        # SAVE_DIR = "results/output_dirs/cifar_coloring/"
        SAVE_DIR = self.args.image_folder

        # Grab the first two samples from MNIST
        dataset = CIFAR10(os.path.join(self.args.run, 'datasets', 'cifar10'), train=False, download=True)
        data = dataset.data.transpose(0, 3, 1, 2)

        for iteration in range(500):
            print("Iteration {}".format(iteration))

            curr_dir = os.path.join(SAVE_DIR, "{:07d}".format(iteration))
            if not os.path.exists(curr_dir):
                os.makedirs(curr_dir)

            rand_idx = np.random.randint(0, data.shape[0] - 1, BATCH_SIZE)
            image = torch.tensor(data[rand_idx, :].astype(np.float) / 255.).float()

            # GT color images
            image_grid = make_grid(image, nrow=GRID_SIZE)
            save_image(image_grid, os.path.join(curr_dir, "gt.png"))

            # Grayscale image
            image_gray = image.mean(1).view(BATCH_SIZE, 1, 32, 32)
            image_grid = make_grid(image_gray, nrow=GRID_SIZE)
            save_image(image_grid, os.path.join(curr_dir, "grayscale.png"))

            image_gray = image_gray.cuda()
            x = nn.Parameter(torch.Tensor(BATCH_SIZE, 3, 32, 32).uniform_()).cuda()

            step_lr=0.00002

            # Noise amounts
            sigmas = np.array([1., 0.59948425, 0.35938137, 0.21544347, 0.12915497,
                           0.07742637, 0.04641589, 0.02782559, 0.01668101, 0.01])
            n_steps_each = 100
              # Weight to put on reconstruction error vs p(x)

            for idx, sigma in enumerate(sigmas):
                lambda_recon = 1.5 / (sigma ** 2)
                # Not completely sure what this part is for
                labels = torch.ones(1, device=x.device) * idx
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2

                
                for step in range(n_steps_each):
                    noise_x = torch.randn_like(x) * np.sqrt(step_size * 2)
                    grad_x = scorenet(x, labels).detach()

                    recon_loss = (torch.norm(torch.flatten(x.mean(1).view(BATCH_SIZE, 1, 32, 32) - image_gray)) ** 2)
                    print(recon_loss)
                    recon_grads = torch.autograd.grad(recon_loss, [x])

                    x = x + (step_size * grad_x) + (-step_size * lambda_recon * recon_grads[0].detach()) + noise_x

            # Write x
            image_grid = make_grid(x, nrow=GRID_SIZE)
            save_image(image_grid, os.path.join(curr_dir, "x.png"))


        import pdb
        pdb.set_trace()

