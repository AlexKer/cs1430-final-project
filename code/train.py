from emotion_dataloader import get_data_dl
import matplotlib.pyplot as plt
from model import ClassificationModel, ViTClassifier, get_PCA_mat
from torcheval.metrics import MulticlassAccuracy
from torcheval.metrics.functional import multiclass_f1_score
import torch
from tqdm import tqdm
import torch.nn as nn
import torchvision.transforms.functional as func

"""
Trains the selected model from model.py
"""

# pca = get_PCA_mat() # uncomment for PCA model
batch_size = 128
epochs = 100
train = get_data_dl(batch_size, True)
test = get_data_dl(batch_size, False)
class_model = ClassificationModel().cuda() # uncomment for CNN model
# class_model = ViTClassifier(7).cuda() # uncomment for ViTB12 model
flatten = nn.Flatten()
# class_model = nn.Sequential( # uncomment for PCA model
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
optimizer = torch.optim.Adam(params=class_model.parameters(), lr=0.05) # Can change lr to change learning rate
n_samples = 7178
n_classes = 7
# Uncomment line below for weighted classification
# weight_class = torch.Tensor([n_samples/(n_classes*958),n_samples/(n_classes*111),n_samples/(n_classes*1024),n_samples/(n_classes*1774),n_samples/(n_classes*1233),n_samples/(n_classes*1247),n_samples/(n_classes*831)]).cuda()
loss_fn = torch.nn.CrossEntropyLoss() # add weight=weight_class as argument for weighted classification
fname = 'CNN.txt' # Log file name
metric = MulticlassAccuracy()
for cur_epoch in tqdm(range(epochs)): # One epoch loop
    metric = MulticlassAccuracy()
    all_loss = 0
    # Train loop
    for i, (images, labels) in tqdm(enumerate(train)):
        optimizer.zero_grad()
        # images = flatten(images) # uncomment for PCA model
        # images = pca.transform(images.numpy()) # uncomment for PCA model
        # images = torch.Tensor(images) # uncomment for PCA model
        images = images.cuda()
        labels = labels.cuda()
        pred = class_model.forward(images)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
        all_loss += loss.item()/(len(train))
    message = 'Epoch: ' + str(cur_epoch) + ' of ' + str(epochs) + ' with loss: ' + str(all_loss)
    print("Train " + str(all_loss))
    with torch.no_grad():
        # Evaluation loop
        val_loss = 0
        for i, (image, labels) in enumerate(test):
            # image = flatten(image) # uncomment for PCA model
            # image = pca.transform(image.numpy()) # uncomment for PCA model
            # image = torch.Tensor(image) # uncomment for PCA model
            image = image.cuda()
            labels = labels.cuda()
            pred = class_model.forward(image)
            loss = loss_fn(pred, labels)
            metric.update(pred, labels)
            val_loss += loss.item()/(len(test))
    acc = metric.compute()
    print("Val " + str(val_loss) + " accuracy: " + str(acc.item()))
    message += ' ' + 'Val: ' + str(val_loss) + ' accuracy ' + str(acc.item()) + '\n'
    with open(fname, 'a+') as f:
        f.write(message)
        f.close()

torch.save(class_model.state_dict(), '/CNN.pth') # Can change second parameter to change saved state dict name