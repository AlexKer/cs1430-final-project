import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from model import VGGModel
from torcheval.metrics import MulticlassAccuracy
from tqdm import tqdm
from torch.optim import Adam  # Import Adam optimizer

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data directories
data_dir = '../data'

# Transformations
train_tfms = tt.Compose([tt.Grayscale(num_output_channels=1),
                         tt.RandomHorizontalFlip(),
                         tt.RandomRotation(degrees=(0, 30)),
                         tt.ToTensor()])
valid_tfms = tt.Compose([tt.Grayscale(num_output_channels=1),
                         tt.ToTensor()])

# Datasets
train_ds = ImageFolder(data_dir + '/train', transform=train_tfms)
valid_ds = ImageFolder(data_dir + '/test', transform=valid_tfms)

# Data loaders
batch_size = 128
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, pin_memory=True)

# Model
model = VGGModel(batch_size).to(device)

# Optimizer
optimizer = Adam(model.parameters(), lr=0.001)

# Metrics and file setup
metric = MulticlassAccuracy()
fname = 'VGG_model.txt'

# Training loop
epochs = 50
for cur_epoch in tqdm(range(epochs), desc="Training Epochs"):
    metric.reset()
    train_loss = 0.0
    
    model.train()
    for images, labels in tqdm(train_dl, desc=f"Epoch {cur_epoch+1} Training Batches", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        outputs = outputs.float()  # Ensure outputs are float32
        loss = model.loss_fn(outputs, labels)
        if torch.isnan(loss):
            print("Nan loss detected")
            print("Outputs:", outputs)
            print("Labels:", labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()
        train_loss += loss.item() / len(train_dl)
    
    print(f"Epoch {cur_epoch+1}/{epochs}, Training Loss: {train_loss:.4f}")

    # Validation loop
    model.eval()
    validation_loss = 0.0
    with torch.no_grad():
        for images, labels in tqdm(valid_dl, desc=f"Epoch {cur_epoch+1} Validation Batches", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs.float()  # Ensure outputs are float32
            loss = model.loss_fn(outputs, labels)
            metric.update(outputs, labels)
            validation_loss += loss.item() / len(valid_dl)
    
    accuracy = metric.compute().item()
    print(f"Validation Loss: {validation_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # Save training and validation results to file
    message = f"Epoch: {cur_epoch+1} of {epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}, Accuracy: {accuracy:.2f}%\n"
    with open(fname, 'a+') as f:
        f.write(message)

    # Save checkpoint every 10 epochs
    if (cur_epoch + 1) % 10 == 0:
        checkpoint_path = f'checkpoint_epoch_{cur_epoch+1}.pth.tar'
        torch.save({
            'epoch': cur_epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),  # Save Adam optimizer state
            'loss': train_loss,
        }, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

# Save the final model
final_model_path = '/mnt/c/Users/rdeme/Documents/Brown/CSCI_1430_Computer_Vision/Project/cs1430-final-project/final_classification_model.pth'
torch.save(model.state_dict(), final_model_path)
print(f"Final model saved at {final_model_path}")
