import torch
import torch.nn as nn
# from vit_pytorch import ViT
from emotion_dataloader import get_data_dl
import os
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

def load_images_from_folder(folder):
    """
    Loads all images inside a folder into a single numpy array
    Args:
        folder (string): path to folder

    Returns:
        numpy.array: array containing all images in the folder 
    """
    images = []
    for emotion in os.listdir(folder):
        emotion_folder = os.path.join(folder, emotion)
        for filename in os.listdir(emotion_folder):
            img_path = os.path.join(emotion_folder, filename)
            img = Image.open(img_path).convert('L')  # convert image to grayscale
            img = np.array(img)
            images.append(img.flatten())
    return np.array(images)

class ClassificationModel(nn.Module):
    def __init__(self) -> None:
        super(ClassificationModel, self).__init__()
        
        self.model = nn.Sequential( 
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Flatten(),
            
            nn.Linear(128 * 6 * 6, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.Linear(64, 7),
            nn.Softmax()
        )
        
    def forward(self, images):
        pred = self.model(images)
        return pred
    
class ViTClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ViTClassifier, self).__init__()
        self.vit = ViT(
            image_size = 48,
            patch_size = 4,
            num_classes = num_classes,
            dim = 512,
            depth = 2,
            heads = 2,
            mlp_dim = 256,
            dropout = 0.1,
            emb_dropout = 0.1
        )

    def forward(self, x):
        return self.vit(x)
    
def get_PCA_mat():
    """
    Calculates the principle component analysis weight matrix for the train and test data.
    The train and test data must have the following structure:
    data
        train
            angry, disgusted, fearful, happy, neutral, sad, surprised
        test
            angry, disgusted, fearful, happy, neutral, sad, surprised
    
    Returns:
        sklearn.decomposition.PCA: returns PCA which transforms inputs into the first 50 eigenvalues
    """
    test = load_images_from_folder('data/test')
    train = load_images_from_folder('data/train')
    array = np.concatenate([train, test])
    pca = PCA(n_components=50).fit(array) 
    
    return pca
