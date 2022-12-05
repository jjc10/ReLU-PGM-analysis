import torch
import torchvision


def get_data(config_dict):
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
