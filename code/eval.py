from torcheval.metrics import MulticlassAccuracy
from torcheval.metrics.functional import multiclass_f1_score
import torch
from emotion_dataloader import get_data_dl
from model import ClassificationModel, ViTClassifier, get_PCA_mat
import torch.nn as nn

batch_size = 128
test = get_data_dl(batch_size, False)
pca = get_PCA_mat()
# class_model = ClassificationModel().cuda()
# class_model = ViTClassifier(7).cuda()
flatten = nn.Flatten()
class_model = nn.Sequential(
    nn.Linear(50, 50),
    nn.BatchNorm1d(50),
    nn.ReLU(),
    
    nn.Linear(50, 50),
    nn.BatchNorm1d(50),
    nn.ReLU(),
            
    nn.Linear(50, 7),
    nn.Softmax(dim=1)
).cuda()
softmax = torch.nn.Softmax()
class_model.load_state_dict(torch.load('/mnt/c/Users/rdeme/Documents/Brown/CSCI_1430_Computer_Vision/Project/cs1430-final-project/PCA.pth'))
all_pred = torch.empty()
all_label = torch.empty()
with torch.no_grad():
    val_loss = 0
    for i, (image, labels) in enumerate(test):
        image = flatten(image)
        image = pca.transform(image.numpy())
        image = torch.Tensor(image).cuda()
        labels = labels.cuda()
        pred = class_model.forward(image)
        all_pred = torch.cat([all_pred, pred.cpu().detach()], dim=0)
        all_label = torch.cat([all_label, labels.cpu().detach()], dim=0)
        

