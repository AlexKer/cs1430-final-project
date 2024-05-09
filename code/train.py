from emotion_dataloader import get_data_dl
import matplotlib.pyplot as plt
from model import VGGModel
from torcheval.metrics import MulticlassAccuracy
import torch
from tqdm import tqdm


batch_size = 128
epochs = 100
train = get_data_dl(batch_size, True)
test = get_data_dl(batch_size, False)
# device = 'cuda' if torch.cuda.is_available() else 'cpu' # check if GPU is available
# class_model = VGGModel(batch_size)
device = 'cpu'  # Use CPU for local run
class_model = VGGModel(batch_size, device=device)  # Initialize model with CPU device
fname = 'VGG_model.txt'
metric = MulticlassAccuracy()

for cur_epoch in tqdm(range(epochs), desc="Epoch Progress"):
    metric = MulticlassAccuracy()
    all_loss = 0
    
    for i, (images, labels) in tqdm(enumerate(train), total=len(train), desc=f"Training Epoch {cur_epoch+1}"):
        class_model.optimizer.zero_grad()
        # image = image.to(class_model.device)
        # labels = labels.to(class_model.device)
        image = images.to(device)  # Move image to CPU
        labels = labels.to(device)  # Move labels to CPU
        pred = class_model.forward(images)
        loss = class_model.loss_fn(pred, labels)
        loss.backward()
        class_model.optimizer.step()
        all_loss += loss.item()/(len(train))
        
    message = 'Epoch: ' + str(cur_epoch) + ' of ' + str(epochs) + ' with loss: ' + str(all_loss)
    print("Train " + str(all_loss))
    
    with torch.no_grad():
        val_loss = 0
        for i, (image, labels) in tqdm(enumerate(test), total=len(test), desc=f"Validation Epoch {cur_epoch+1}"):
            # image = image.to(class_model.device)
            # labels = labels.to(class_model.device)
            image = images.to(device)  # Move image to CPU
            labels = labels.to(device)  # Move labels to CPU
            pred = class_model.forward(image)
            loss = class_model.loss_fn(pred, labels)
            # metric.update(pred, labels)
            metric.update(pred.argmax(dim=1), labels)
            val_loss += loss.item()/(len(test))
    acc = metric.compute()
    
    print("Val " + str(val_loss) + " accuracy: " + str(acc.item()))
    message += ' ' + 'Val: ' + str(val_loss) + ' accuracy ' + str(acc.item()) + '\n'
    
    with open(fname, 'a+') as f:
        f.write(message)
        f.close()

# torch.save(class_model.state_dict(), '/home/soh62/CS1430-CV-Project/')
torch.save(class_model.state_dict(), './VGG_model.pth')  #