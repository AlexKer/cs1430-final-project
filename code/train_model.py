import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from torch.utils.data import DataLoader
from model import VGGModel 
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data directories
data_dir = '../data' 

# Transformations
train_tfms = tt.Compose([tt.Grayscale(num_output_channels=1),
                         tt.RandomHorizontalFlip(),
                        #  tt.RandomVerticalFlip(p=0.5),
                         tt.RandomRotation(degrees=(0, 30)),
                        #  tt.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2)),
                        #  tt.RandomPerspective(distortion_scale=0.5, p=0.5),
                        #  tt.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
                         tt.ToTensor() # convert image to tensor
                         ])
valid_tfms = tt.Compose([tt.Grayscale(num_output_channels=1),
                         tt.ToTensor()
])

# Datasets
train_ds = ImageFolder(data_dir + '/train', transform=train_tfms)
valid_ds = ImageFolder(data_dir + '/test', transform=valid_tfms)

# Data loaders
batch_size = 50
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True, pin_memory=True)

# Model
model = VGGModel(learning_rate=0.001).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to save the model checkpoint
def save_checkpoint(epoch, model, optimizer, filename='checkpoint.pth.tar'):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, filename)
    print(f"Saved checkpoint: {filename} at epoch {epoch}")
    

# Training loop
# Help used from HW5 and ChatGPT and DL1470 homework
def train(model, train_dl, valid_dl, epochs, save_interval=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_dl, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Calculate average training loss for the current epoch
        avg_train_loss = running_loss / len(train_dl)

        # Switch to evaluation mode for validation
        model.eval()
        validation_loss = 0.0
        total, correct = 0, 0
        with torch.no_grad():
            for images, labels in valid_dl:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Calculate average validation loss and accuracy for the current epoch
        avg_val_loss = validation_loss / len(valid_dl)
        accuracy = 100 * correct / total

        # Print training and validation results for the current epoch
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

        # Save checkpoint every 'save_interval' epochs
        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            save_checkpoint(epoch + 1, model, optimizer, f'checkpoint_epoch_{epoch+1}.pth.tar')

train(model, train_dl, valid_dl, epochs=50)

# Optionally save the final model
torch.save(model.state_dict(), 'final_emotion_vgg_model.pth')
