from emotion_dataloader import get_data_dl
import matplotlib.pyplot as plt
from model import ClassificationModel
from torcheval.metrics import MulticlassAccuracy
import torch
from tqdm import tqdm


batch_size = 128
epochs = 100
train = get_data_dl(batch_size, True)
test = get_data_dl(batch_size, False)
class_model = ClassificationModel(batch_size)
fname = 'cnn_model.txt'
metric = MulticlassAccuracy()
for cur_epoch in tqdm(range(epochs)):
    metric = MulticlassAccuracy()
    all_loss = 0
    
    for i, (images, labels) in enumerate(train):
        class_model.optimizer.zero_grad()
        images = images.cuda()
        labels = labels.cuda()
        pred = class_model.forward(images)
        loss = class_model.loss_fn(pred, labels)
        loss.backward()
        class_model.optimizer.step()
        all_loss += loss.item()/(len(train))
    message = 'Epoch: ' + str(cur_epoch) + ' of ' + str(epochs) + ' with loss: ' + str(all_loss)
    print("Train " + str(all_loss))
    with torch.no_grad():
        val_loss = 0
        for i, (image, labels) in enumerate(test):
            image = image.to(class_model.device)
            labels = labels.to(class_model.device)
            pred = class_model.forward(image)
            loss = class_model.loss_fn(pred, labels)
            metric.update(pred, labels)
            val_loss += loss.item()/(len(test))
    acc = metric.compute()
    print("Val " + str(val_loss) + " accuracy: " + str(acc.item()))
    message += ' ' + 'Val: ' + str(val_loss) + ' accuracy ' + str(acc.item()) + '\n'
    with open(fname, 'a+') as f:
        f.write(message)
        f.close()

torch.save(class_model.state_dict(), '/home/rdemello/CSCI1430/')