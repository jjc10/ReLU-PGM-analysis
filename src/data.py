import torch
import torchvision

import os
import glob
import pandas as pd
from torchvision import transforms
from PIL import Image
# import tensorflow_datasets as tfds

def get_mnist_data(config_dict):
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('.', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (MNIST_MEAN,), (MNIST_STD,))
                                   ])),
        batch_size=config_dict['batch_size_train'], shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('.', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (MNIST_MEAN,), (MNIST_STD,))
                                   ])),
        batch_size=config_dict['batch_size_test'], shuffle=True)

    return train_loader, test_loader

class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, directory, transform = None, subset = None):
        self.directory = directory
        if transform:
            self.transform = transform
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.transform = transforms.Compose([
                torchvision.transforms.Grayscale(num_output_channels=3), # there are a few greyscale images
                transforms.RandomResizedCrop((500, 300)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        self.labels = glob.glob1(self.directory, "*.JPEG")
        self.labels = list(map(lambda n: n.replace('.JPEG', ''), self.labels))
        self.subset = subset

    def __len__(self):
        return self.subset if self.subset else len(self.labels)
    def __getitem__(self, idx):
        image = Image.open(f'{self.directory}/{self.labels[idx]}.JPEG')
        image_processed = self.transform(image)
        return image_processed, self.labels[idx]

def get_imagenet_data(config_dict, directory):
    pass

def get_imagenet_test_data(config_dict, directory, subset = None):
    dataset = ImageNetDataset(directory, subset = subset)
    return torch.utils.data.DataLoader(dataset, batch_size=config_dict['batch_size_test'], shuffle=True)
