import torch
import os

FIGURE_FOLDER = 'figures'
RESULTS_FOLDER = 'result_storage'


def get_config():

    config_dict = {'n_epochs': 4,
                   'batch_size_train': 64,
                   'batch_size_test': 100,
                   'learning_rate': 0.01,
                   'momentum': 0.9,
                   'depth': 2,
                   'hidden_size': 6,
                   'trials': 1}

    return config_dict

def get_cifar_config():
    config_dict = {'n_epochs': 4,
                   'batch_size_train': 64,
                   'batch_size_test': 100,
                   'learning_rate': 0.01,
                   'momentum': 0.9,
                   'depth': 2,
                   'hidden_size': 8,
                   'trials': 1}
    return config_dict

def get_pretrained_imagenet_config():
    config_dict = {
                   'batch_size_test': 100,
                   'momentum': 0.9,
                   'trials': 1
    }

    return config_dict

def set_up_paths(folders):
    for folder in folders:
        if not os.path.exists(folder):
            print('Creating folder', folder, '...')
            os.makedirs(folder)


def set_randomness(seed):
    random_seed = seed
    torch.manual_seed(random_seed)
