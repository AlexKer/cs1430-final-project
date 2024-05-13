from emotion_dataloader import get_data_dl
import matplotlib.pyplot as plt
from model import ModifiedVGGModel
from torcheval.metrics import MulticlassAccuracy
import torch
from tqdm import tqdm

batch_size = 128
epochs = 2
train = get_data_dl(batch_size, True)
test = get_data_dl(batch_size, False)
class_model = ModifiedVGGModel().cuda()
# class_model = ModifiedVGGModel()
optimizer = torch.optim.Adam(params=class_model.parameters(), lr=0.05)
loss_fn = torch.nn.CrossEntropyLoss()
fname = 'modified_vgg_model.txt'
metric = MulticlassAccuracy()

for cur_epoch in tqdm(range(epochs), desc="Epoch Progress"):
    metric = MulticlassAccuracy()
    all_loss = 0
    train_progress = tqdm(train, desc=f"Training Epoch {cur_epoch+1}/{epochs}")
    
    for i, (images, labels) in enumerate(train_progress):
        optimizer.zero_grad()
        images = images.cuda()
        labels = labels.cuda()
        pred = class_model(images)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()
        all_loss += loss.item()/(len(train))
        train_progress.set_postfix(loss=all_loss)
    message = 'Epoch: ' + str(cur_epoch) + ' of ' + str(epochs) + ' with loss: ' + str(all_loss)
    print("Train " + str(all_loss))
    with torch.no_grad():
        val_loss = 0
        test_progress = tqdm(test, desc=f"Validation Epoch {cur_epoch+1}/{epochs}")
        for i, (image, labels) in enumerate(test_progress):
            image = image.cuda()
            labels = labels.cuda()
            pred = class_model(image)
            loss = loss_fn(pred, labels)
            metric.update(pred, labels)
            val_loss += loss.item()/(len(test))
            test_progress.set_postfix(val_loss=val_loss)
    if cur_epoch == 10:
        torch.save(class_model.state_dict(), '/home/soh62/CS1430-CV-Project/cs1430-final-project/code/VGG_model_2.pth')
    acc = metric.compute()
    print("Val " + str(val_loss) + " accuracy: " + str(acc.item()))
    message += ' ' + 'Val: ' + str(val_loss) + ' accuracy ' + str(acc.item()) + '\n'
    with open(fname, 'a+') as f:
        f.write(message)
        f.close()

torch.save(class_model.state_dict(),'./modified_vgg_model_final.pth') 


