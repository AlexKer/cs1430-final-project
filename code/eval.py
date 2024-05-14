from torcheval.metrics import MulticlassAccuracy, MulticlassConfusionMatrix
from torcheval.metrics.functional import multiclass_f1_score
import torch
from emotion_dataloader import get_data_dl
from model import ClassificationModel, ViTClassifier, get_PCA_mat
import torch.nn as nn
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torchsummary import summary

"""
Generates confusion matrix and calculates F1 score for chosen model loaded from .pth file
"""

batch_size = 128
test = get_data_dl(batch_size, False)
# pca = get_PCA_mat() # uncomment for PCA classifier model
class_model = ClassificationModel().cuda() # uncomment for CNN model
# summary(class_model, (3,48,48)) # uncomment for model summary
# class_model = ViTClassifier(7).cuda() # uncomment for ViTB12 model
# summary(class_model, (3,48,48)) # uncomment for model summary
flatten = nn.Flatten()
# class_model = nn.Sequential( # uncomment for PCA classifier model
#     nn.Linear(50, 50),
#     nn.BatchNorm1d(50),
#     nn.ReLU(),
    
#     nn.Linear(50, 50),
#     nn.BatchNorm1d(50),
#     nn.ReLU(),
            
#     nn.Linear(50, 7),
#     nn.Softmax(dim=1)
# ).cuda()
softmax = torch.nn.Softmax()
class_model.load_state_dict(torch.load('../weights/CNN.pth')) # Replace with model weights
all_pred = torch.empty(0)
all_label = torch.empty(0)
with torch.no_grad():
    val_loss = 0
    for i, (image, labels) in enumerate(test):
        # image = flatten(image) # uncomment for PCA classifier model
        # image = pca.transform(image.numpy()) # uncomment for PCA classifier model
        # image = torch.Tensor(image) # uncomment for PCA classifier model
        image = image.cuda() 
        labels = labels.cuda()
        pred = class_model.forward(image)
        all_pred = torch.cat([all_pred, pred.cpu().detach()], dim=0)
        all_label = torch.cat([all_label, labels.cpu().detach()], dim=0)


all_pred = torch.argmax(all_pred, dim=1)
f1_score = multiclass_f1_score(all_pred, all_label, num_classes=7)
print("F1 score ", f1_score)
cm = confusion_matrix(all_label.numpy(), all_pred.numpy())
ConfusionMatrixDisplay(cm, display_labels=['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']).plot()
plt.show()