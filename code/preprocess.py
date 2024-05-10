"""
Homework 5 - CNNs
CS1430 - Computer Vision
Brown University
"""

import os
import random
import numpy as np
from PIL import Image
import torch
from torchvision import transforms, datasets

import hyperparameters as hp

class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path, task):

        self.data_path = data_path
        self.task = task

        # Dictionaries for (label index) <--> (class name)
        self.idx_to_class = {}
        self.class_to_idx = {}

        # For storing list of classes
        self.classes = [""] * hp.num_classes

        # Mean and std for standardization
        self.mean = np.zeros((3, hp.img_size, hp.img_size))
        self.std = np.ones((3, hp.img_size, hp.img_size))
        self.calc_mean_and_std()

        # Setup data generators
        # These feed data to the training and testing routine based on the dataset
        self.train_data = self.get_data(
            "train", task == '3', True, True)
        self.test_data = self.get_data(
            "test", task == '3', False, False)
        # self.test_data = self.get_data("/content/homework5_cnns-glitterer/data/test/", task == '3', False, False)

    def calc_mean_and_std(self):
        """ Calculate mean and standard deviation of a sample of the
        training dataset for standardization.

        Arguments: none

        Returns: none
        """

        # Get list of all images in training directory
        file_list = [os.path.join(dp, f) for dp, dn, filenames in os.walk(os.path.join(self.data_path, "train")) for f in filenames if f.endswith('.png')]
        random.shuffle(file_list)
        file_list = file_list[:hp.preprocess_sample_size]
        data_sample = np.zeros((len(file_list), 3, hp.img_size, hp.img_size))
        for i, file_path in enumerate(file_list):
            img = Image.open(file_path).convert('RGB')
            img = img.resize((hp.img_size, hp.img_size))
            img = np.array(img, dtype=np.float32).transpose((2, 0, 1)) / 255.
            data_sample[i] = img
        self.mean = np.mean(data_sample, axis=0)
        self.std = np.std(data_sample, axis=0)

    def standardize(self, img):
        """ Function for applying standardization to an input image.

        Arguments:
            img - numpy array of shape (image size, image size, 3)

        Returns:
            img - numpy array of shape (image size, image size, 3)
        """

        # TASK 1
        # TODO: Standardize the input image. Use self.mean and self.std
        #       that were calculated in calc_mean_and_std() to perform
        #       the standardization.
        # =============================================================

        img = (img - self.mean) / self.std     # replace this code
        # =============================================================

        return img

    def preprocess_fn(self, img):
        """ Preprocess function for ImageDataGenerator. """

        if self.task == '3':
            preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: self.standardize(x))
            ])
        return preprocess(img)

    def custom_preprocess_fn(self, img):
        """ Custom preprocess function for ImageDataGenerator. """

        if self.task == '3':
            preprocess = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            preprocess = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: self.standardize(x))
            ])


        if random.random() < 0.3:
            img = img + torch.tensor(np.random.uniform(-0.1, 0.1, size=img.shape), dtype=torch.float32)

        return img

    def get_data(self, folder, is_vgg, shuffle, augment):
        """ Returns an image data generator which can be iterated
        through for images and corresponding class labels.

        Arguments:
            path - Filepath of the data being imported, such as
                   "../data/train" or "../data/test"
            is_vgg - Boolean value indicating whether VGG preprocessing
                     should be applied to the images.
            shuffle - Boolean value indicating whether the data should
                      be randomly shuffled.
            augment - Boolean value indicating whether the data should
                      be augmented or not.

        Returns:
            An iterable image-batch generator
        """

        path = os.path.join(self.data_path, folder)
        if augment:
            data_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.RandomResizedCrop(hp.img_size if not is_vgg else 224, scale=(0.8, 1.0)),
                self.preprocess_fn
            ])
        else:
            data_transforms = self.preprocess_fn

        dataset = datasets.ImageFolder(root=path, transform=data_transforms)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=hp.batch_size, shuffle=shuffle)
        return data_loader

