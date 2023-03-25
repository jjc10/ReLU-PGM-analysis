import matplotlib.pyplot as plt
import torch
import torchvision

import os
from dataclasses import dataclass, field
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


@dataclass
class ImageNetDatasetSpec:
    """Dataclass for specifying what th Imagenet dataset should load"""
    number_of_classes: int = 10
    samples_per_class:int = 50
    """overrides number_of_classes"""
    classes: list[str] = field(default_factory=list)

    def num_of_classes(self):
        return len(self.classes) if len(self.classes) > 0 else self.number_of_classes

class ImageNetDataset(torch.utils.data.Dataset):
    """Dataset for imagenet.

    The directory should point to the directory containing imagenet and also specify whether to use the training data or validation data
    by appending /train or /val.
    ex: "/Users/joudchataoui/code/McGill/networks-lab/relu_code/imagenet/imagenette2/train"

    transform specifies the torch vision transform to apply on images.

    image_net_dataset_spec is a data object for narrowing the behaviour for example the number of classes or number of instances per class
    """
    def __init__(self, directory, transform = None, image_net_dataset_spec: ImageNetDatasetSpec = None):
        self.directory = directory
        if transform:
            self.transform = transform
        else:
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.transform = transforms.Compose([
                torchvision.transforms.Grayscale(num_output_channels=3), # there are a few greyscale images
                transforms.Resize((300, 500)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        if image_net_dataset_spec:
            self.classes, self.image_list = self._handle_imagenet_dataset_spec(image_net_dataset_spec)
        else:
            self.classes = glob.glob(f"{self.directory}/*")
            self.classes = [c.split('/')[-1] for c in self.classes]
            self.image_list = []
            for c in self.classes:
                self.image_list.extend(glob.glob(f"{self.directory}/{c}/*.JPEG"))

    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        image = Image.open(self.image_list[idx])
        label = self._get_class_label_for_file_name(self.image_list[idx])
        image_processed = self.transform(image)
        return image_processed, label

    def _get_class_label_for_file_name(self, file_name):
        return file_name.split('/')[-2] # last item is the file name (format is label_number.jpeg), second to last is the label
    def _handle_imagenet_dataset_spec(self, imagenet_dataset_spec):
        if len(imagenet_dataset_spec.classes) > 0:
            classes = imagenet_dataset_spec.classes
        else:
            classes = glob.glob(f"{self.directory}/*")[:imagenet_dataset_spec.number_of_classes]
            classes = [c.split('/')[-1] for c in classes]
        image_list = []
        for c in classes:
            image_list_for_class = glob.glob(f"{self.directory}/{c}/*.JPEG")[:imagenet_dataset_spec.samples_per_class]
            image_list.extend(image_list_for_class)
        return classes, image_list

def get_imagenet_data(config_dict, directory, train_dataset_spec = None, test_dataset_spec = None):
    train_ds = ImageNetDataset(f"{directory}/train", image_net_dataset_spec = train_dataset_spec)
    test_ds = ImageNetDataset(f"{directory}/test", image_net_dataset_spec = test_dataset_spec)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config_dict['batch_size_train'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config_dict['batch_size_test'], shuffle=True)
    return train_loader, test_loader

def get_imagenet_test_data(config_dict, directory, subset = None, image_list = []):
    dataset = ImageNetDataset(directory, subset = subset, image_list=image_list)
    return torch.utils.data.DataLoader(dataset, batch_size=config_dict['batch_size_test'], shuffle=True)
