import torch
import torch.nn as nn
import torchvision.models as models
from emotion_dataloader import get_data_dl



class ClassificationModel(nn.Module):
    def __init__(self, batch_size:int, device = 'cuda', lr=0.09) -> None:
        super(ClassificationModel, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
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
        ).to(device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        
    def forward(self, images):
        pred = self.model(images)
        return pred
    