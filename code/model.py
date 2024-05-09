import torch
import torch.nn as nn
import torchvision.models as models
from emotion_dataloader import get_data_dl



class ClassificationModel(nn.Module):
    def __init__(self, batch_size:int, device = 'cpu', lr=0.09) -> None:
        super(ClassificationModel, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.model = nn.Sequential(
            
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # # Block 5
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),

           
            nn.Flatten(),

            # nn.Linear(512, 120),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(120, 64),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(64, 7),
            # nn.Softmax(dim=1)
            
            nn.Linear(512 * 3 * 3, 4096),  # 调整为最终特征图尺寸3x3
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 7),
            nn.Softmax(dim=1)
            
        ).to(device)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
        
    def forward(self, images):
        pred = self.model(images)
        return pred
    