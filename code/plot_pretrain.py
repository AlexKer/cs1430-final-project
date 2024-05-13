import matplotlib.pyplot as plt

cwd = '/mnt/c/Users/rdeme/Documents/Brown/CSCI_1430_Computer_Vision/Project/cs1430-final-project/'
log_files = [cwd+'CNN.txt', cwd+'PCA.txt', cwd+'vision_transformer.txt', cwd+'vision_transformer_weight.txt']

plt.figure(figsize=(12, 8))

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