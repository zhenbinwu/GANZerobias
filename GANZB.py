#!/usr/bin/env python3
# encoding: utf-8

# File        : GANZB.py
# Author      : Zhenbin Wu
# Contact     : zhenbin.wu@gmail.com
# Date        : 2018 Mar 22
#
# Description : 


import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
print(torch.__version__)

class L1Jets(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, csv_file, root_dir, transform=None):
        super().__init__()
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.jets = np.load("emulator.npy")
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.jets)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:].as_matrix()
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

BATCH_SIZE = 1
dataset = np.load("emulator.npy").astype(np.float32)
unpacked = np.load("unpack.npy").astype(np.float32)
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=BATCH_SIZE, 
                                          shuffle=True,
                                          num_workers=3)

n_batches = int(np.ceil(len(dataset)/ BATCH_SIZE)) # 60000 / 100
from torch import nn


# `torch.nn.Sequential` is a container that allows you to specify Modules to be added to your model in the order they are passed into the constructor. Alternatively, you can also pass them as an `OrderedDict`.

# In[12]:

# Discriminator
D = nn.Sequential(
    nn.Linear(3, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

# Generator 
G = nn.Sequential(
    nn.Linear(3, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 256),
    nn.LeakyReLU(0.2),
    nn.Linear(256, 3),
    nn.Tanh()
)

# if you're running on GPU, move models there
if torch.cuda.is_available():
    D.cuda()
    G.cuda()


# Define your loss function and optimizer by picking from the ones already available in `torch.nn` or defining your own
criterion = nn.BCELoss() # Or remove nn.Sigmoid() and use 
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)


from torch.autograd import Variable

def to_var(x):
    # first move to GPU, if necessary
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# In[15]:

import torch.nn.functional as F
from torchvision.utils import save_image

def denorm(x):
    # convert back from [-1, 1] to [0, 1]
    out = (x + 1) / 2
    return out #out.clamp(0, 1)

N_EPOCHS = 200

try: # allow for manual keyboard interrupt
    # loop through epochs
    for epoch in range(N_EPOCHS):
        # loop through batches (no need for class labels right now)
        for batch_number, images in enumerate(data_loader): 
            # Reshape training dataset images from (batch_size, 28, 28) to (batch_size, 28*28) for 
            # processing through fully-connected net 
            batch_size = images.shape[0] # this specific batch size (last one may not be equal to BATCH_SIZE)
            images = to_var(images.view(batch_size, -1))

            # Create targets for the discriminator network D
            # (can use label flipping or label smoothing)
            real_labels = to_var(torch.ones(batch_size, 1)) 
            fake_labels = to_var(torch.zeros(batch_size, 1))

            # 1) TRAIN DISCRIMINATOR
            # Evaluate the discriminator on the real input images
            outputs = D(images) # or D(add_instance_noise(images))
            real_score = outputs
            # Compute the discriminator loss with respect to the real labels (1s)
            d_loss_real = criterion(outputs, real_labels)

            # Draw random 64-dimensional noise vectors as inputs to the generator network
            # z = to_var(torch.Tensor(unpacked[epoch-1*batch_number:epoch*batch_number])) # the latents space is 64D
            z = to_var(torch.randn(batch_size, 3))
            # Transform the noise through the generator network to get synthetic images
            fake_images = G(z)
            # Evaluate the discriminator on the fake images
            outputs = D(fake_images) # or D(add_instance_noise(fake_images))
            fake_score = outputs
            # Compute the discriminator loss with respect to the fake labels (0s)
            d_loss_fake = criterion(outputs, fake_labels)

            # Backprop + Optimize the discriminator
            d_loss = d_loss_real + d_loss_fake
            D.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # 2) TRAIN GENERATOR
            # Draw random 64-dimensional noise vectors as inputs to the generator network
            z = to_var(torch.randn(batch_size, 3))
            # Transform the noise through the generator network to get synthetic images
            fake_images = G(z)
            # Evaluate the (new) discriminator on the fake images
            outputs = D(fake_images)

            # Compute the cross-entropy loss with "real" as target (1s). This is what the G wants to do
            g_loss = criterion(outputs, real_labels)

            # Backprop + Optimize the generator
            D.zero_grad() # probably unnecessary?
            G.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if (batch_number + 1) % 300 == 0:
                print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, '
                      'g_loss: %.4f, Mean D(x): %.2f, Mean D(G(z)): %.2f' 
                      %(epoch,
                        N_EPOCHS,
                        batch_number + 1,
                        n_batches,
                        d_loss.data[0],
                        g_loss.data[0],
                        real_score.data.mean(),
                        fake_score.data.mean())
                )

        # # Save real images once
        # if (epoch + 1) == 1:
            # images = images.view(images.size(0), 1, 28, 28) # reshape
            # save_image(denorm(images.data), './data/real_images.png')

        # # Save sampled images
        # fake_images = fake_images.view(fake_images.size(0), 1, 28, 28) #reshape
        # save_image(denorm(fake_images.data), './data/fake_images-%0.3d.png' %(epoch + 1))
        
        # Save the trained parameters 
        # torch.save(G.state_dict(), './weights/generator-%0.3d.pkl' %(epoch + 1))
        # torch.save(D.state_dict(), './weights/discriminator-%0.3d.pkl' %(epoch + 1))
        
except KeyboardInterrupt:
    print 'Training ended early.'


# ## Evaluation

# Load back the weights from your favorite epoch.

# In[32]:

G.load_state_dict(torch.load('./weights/generator-final.pkl'))
D.load_state_dict(torch.load('./weights/discriminator-final.pkl'))

