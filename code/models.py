"""
Homework 5 - CNNs
CS1430 - Computer Vision
Brown University
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import hyperparameters as hp


class VGGModel(nn.Module):
    def __init__(self):
        super(VGGModel, self).__init__()

        # Define the feature blocks of the VGG-like architecture
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Define the classifier part of the VGG-like architecture
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 512), # Assumes input size of 224x224
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7),  # Number of classes
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def loss_fn(self, labels, predictions):
        criterion = nn.CrossEntropyLoss()
        return criterion(predictions, labels)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VGGModel().to(device)
print(model)
