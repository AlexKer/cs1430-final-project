import torch
import torch.nn as nn
import torchvision.models as models
from emotion_dataloaderz import get_data_dl



class ClassificationModel(nn.Module):
    def __init__(self, batch_size:int, device='cuda', lr=0.09) -> None:
        super(ClassificationModel, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

           
            nn.Flatten(),

            nn.Linear(512, 120),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(120, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 7),
            nn.Softmax(dim=1)
        ).to(device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        
    def forward(self, images):
        pred = self.model(images)
        return pred