import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

cwd = '/home/soh62/CS1430-CV-Project/cs1430-final-project/'
# cwd = '/Users/seikoh/BrownWorkspace/CS1430_Projects/cs1430-final-project/'

def get_data_dl(batchsize: int, training: bool):
    '''
    Get the training or testing data dataloader for the emotion classification data.
    Inputs: batchsize of the data, training bool (True = training data, False = testing data)
    Outputs: Dataloader with the data put into batches, returns image, label as int.
    '''
    if training:
        train_transforms = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.RandomVerticalFlip(0.3),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            # transforms.Grayscale(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        data = dset.ImageFolder(cwd + 'data/train', transform = train_transforms)
        dl = DataLoader(data, batch_size=batchsize, shuffle=True)
    else:
        test_transforms = transforms.Compose([
            # transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Grayscale()
        ])
        data = dset.ImageFolder(cwd + 'data/test', transform = test_transforms)
        dl = DataLoader(data, batch_size=batchsize, shuffle=True)
    return dl