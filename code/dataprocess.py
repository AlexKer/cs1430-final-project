import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np



# Looking into the directory
# find . -name ".DS_Store" -print -delete to delete all .DS_Store files
data_dir = '../data'
train_classes_list = os.listdir(data_dir + "/train")
test_class_list = os.listdir(data_dir + "/test")
print(f'Train Classes - {train_classes_list}')
print(f'Test Classes - {test_class_list}')


# Data transforms (Gray Scaling & data augmentation)
train_tfms = tt.Compose([tt.Grayscale(num_output_channels=1),
                         tt.RandomHorizontalFlip(),
                        #  tt.RandomVerticalFlip(p=0.5),
                         tt.RandomRotation(degrees=(0, 30)),
                        #  tt.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                        #  tt.RandomPerspective(distortion_scale=0.5, p=0.5),
                        #  tt.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
                         tt.ToTensor() # convert image to tensor
                         ])

valid_tfms = tt.Compose([tt.Grayscale(num_output_channels=1), tt.ToTensor()])

# Emotion Detection datasets
# Load images from a directory to automatically assign labels to each of the class's images
train_ds = ImageFolder(data_dir + '/train', train_tfms)
valid_ds = ImageFolder(data_dir + '/test', valid_tfms)

# PyTorch data Loader
# pin_memory to get faster data transfer to CUDA-enabled GPUs
batch_size = 70
train_dl = DataLoader(train_ds, batch_size, shuffle=True, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size, shuffle=True, pin_memory=True)



# visualize data processing results
def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(6, 6))
        
        grid_images = make_grid(images[:30], nrow=6) # Takes up to the first 30 images from the batch
        permuted_images = grid_images.permute(1, 2, 0) # rearrange dimensions of the tensor to be used for matplotlib (C,H,W) -> (H,W,C)
        images = permuted_images.numpy()
        print(images.shape)

        # If grayscale img, change img dimension to 2d for matplotlib
        if images.shape[2] == 1:
            images = images.squeeze(2)

        ax.imshow(images, cmap='gray')
        plt.show()
        break
show_batch(train_dl)