from emotion_dataloader import get_data_dl
import matplotlib.pyplot as plt
from model import ClassificationModel, ViTClassifier
from torcheval.metrics import MulticlassAccuracy
import torch
from tqdm import tqdm


batch_size = 128
epochs = 100
train = get_data_dl(batch_size, True)
test = get_data_dl(batch_size, False)
# class_model = ClassificationModel().cuda()
class_model = ViTClassifier(7).cuda()
softmax = torch.nn.Softmax()
optimizer = torch.optim.Adam(params=class_model.parameters(), lr=0.05)
n_samples = 7178
n_classes = 7
weight_class = torch.Tensor([n_samples/(n_classes*958),n_samples/(n_classes*111),n_samples/(n_classes*1024),n_samples/(n_classes*1774),n_samples/(n_classes*1233),n_samples/(n_classes*1247),n_samples/(n_classes*831)]).cuda()
loss_fn = torch.nn.CrossEntropyLoss(weight=weight_class)
fname = 'vision_transformer_weight.txt'
metric = MulticlassAccuracy()
for cur_epoch in tqdm(range(epochs)):
    metric = MulticlassAccuracy()
    all_loss = 0
    
    for i, (images, labels) in tqdm(enumerate(train)):
        optimizer.zero_grad()
        images = images.cuda()
        labels = labels.cuda()
        pred = class_model.forward(images)
        pred = softmax(pred)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
        all_loss += loss.item()/(len(train))
    message = 'Epoch: ' + str(cur_epoch) + ' of ' + str(epochs) + ' with loss: ' + str(all_loss)
    print("Train " + str(all_loss))
    with torch.no_grad():
        val_loss = 0
        for i, (image, labels) in enumerate(test):
            image = image.cuda()
            labels = labels.cuda()
            pred = class_model.forward(image)
            pred = softmax(pred)
            loss = loss_fn(pred, labels)
            metric.update(pred, labels)
            val_loss += loss.item()/(len(test))
    if cur_epoch == 10:
        torch.save(class_model.state_dict(), '/home/rdemello/CSCI1430/vision_transformer10_wt.pth')
    acc = metric.compute()
    print("Val " + str(val_loss) + " accuracy: " + str(acc.item()))
    message += ' ' + 'Val: ' + str(val_loss) + ' accuracy ' + str(acc.item()) + '\n'
    with open(fname, 'a+') as f:
        f.write(message)
        f.close()

torch.save(class_model.state_dict(), '/home/rdemello/CSCI1430/vision_transformer_weighted.pth')