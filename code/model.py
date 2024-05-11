import torch
import torch.nn as nn
from vit_pytorch import ViT
from emotion_dataloader import get_data_dl



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
            depth = 4,
            heads = 4,
            mlp_dim = 256,
            dropout = 0.1,
            emb_dropout = 0.1
        )

    def forward(self, x):
        return self.vit(x)