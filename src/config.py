import torch


def get_config():

    config_dict = {'n_epochs': 3,
                   'batch_size_train': 64,
                   'batch_size_test': 100,
                   'learning_rate': 0.01,
                   'momentum': 0.5,
                   'depht': 2,
                   'hidden_size': 8}

    return config_dict


def set_randomness():
    random_seed = 1
    torch.manual_seed(random_seed)
