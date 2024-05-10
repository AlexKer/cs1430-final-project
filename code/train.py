from emotion_dataloader import get_data_dl
import matplotlib.pyplot as plt
from model import ClassificationModel
from torchvision.models import VisionTransformer
from torcheval.metrics import MulticlassAccuracy
import torch
from tqdm import tqdm


batch_size = 128
epochs = 100
train = get_data_dl(batch_size, True)
test = get_data_dl(batch_size, False)
# class_model = ClassificationModel(batch_size)
class_model = VisionTransformer(48, 3, 4, 4, 640, 256, 0.1, 0.1, 7).cuda()
softmax = torch.nn.Softmax()
optimizer = torch.optim.Adam(params=class_model.parameters(), lr=0.02)
loss_fn = torch.nn.CrossEntropyLoss()
fname = 'vision_transformer.txt'
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
            loss = loss_fn(pred, labels)
            metric.update(pred, labels)
            val_loss += loss.item()/(len(test))
    if cur_epoch == 10:
        torch.save(class_model.state_dict(), '/home/rdemello/CSCI1430/vision_transformer10.pth')
    acc = metric.compute()
    print("Val " + str(val_loss) + " accuracy: " + str(acc.item()))
    message += ' ' + 'Val: ' + str(val_loss) + ' accuracy ' + str(acc.item()) + '\n'
    with open(fname, 'a+') as f:
        f.write(message)
        f.close()

torch.save(class_model.state_dict(), '/home/rdemello/CSCI1430/vision_transformer.pth')