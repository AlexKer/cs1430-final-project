## VGG-16 model

- This VGG-16 directory is to be used separately as if used in a new branch
- Uses the assignment hw5 as the base code 
- Changed parameters and some parts of the code for better readability and for it to work on our own dataset

- In order to have it run, paths to dataset and other environments would need to be adjusted according to the person's environment
- Data should be added. Our model uses Kaggle dataset("https://www.kaggle.com/code/odins0n/emotion-detection/input")

- The checkpoints are from running the VGG-16 model on locally, taking around 780minutes. The log and misclassified models are from the highest epoch checkpoint which is the `vgg.weights.e014-acc0.4964.h5`

- Use Google Colab to get the Tensorboard working for graphs and misclassification images from certain checkpoints.
