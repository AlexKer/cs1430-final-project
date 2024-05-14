import matplotlib.pyplot as plt

log_files = ['CNN.txt', 'PCA.txt', 'vision_transformer.txt', 'vision_transformer_weight.txt'] # Can replace with all of your log files 

plt.figure(figsize=(12, 8))
"""
Plots validation loss, train loss and accuracy from a generated log file from train.py
"""
for i, log_file in enumerate(log_files):
    with open(log_file, 'r') as f:
        lines = f.readlines()

    epochs = []
    losses = []
    vals = []
    accs = []

    for line in lines:
        words = line.split()
        
        epoch = int(words[1])
        loss = float(words[6])
        val = float(words[8])
        acc = float(words[10])
        
        epochs.append(epoch)
        losses.append(loss)
        vals.append(val)
        accs.append(acc)

    plt.subplot(3, 1, 1)
    plt.plot(epochs, losses, label=f'Training Loss {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    

    plt.subplot(3, 1, 2)
    plt.plot(epochs, vals, label=f'Validation Loss {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')

    plt.subplot(3, 1, 3)
    plt.plot(epochs, accs, label=f'Testing Accuracy {i+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

plt.legend(['CNN', 'PCA', 'ViT', 'ViT with weighted loss'])
plt.tight_layout()
plt.show()