
import os
import time
import pickle as pk
import torch
from src.config import RESULTS_FOLDER
from src.model import build_model


def generate_run_id(prefix = ''):
    now = int(time.time())
    string_version_now = str(now)
    return f'{prefix}_{string_version_now}'


def store_results(file_name, data_to_store, run_path):
    file_path = os.path.join(run_path, file_name+'.pk')
    with open(file_path, 'wb') as file:
        pk.dump(data_to_store, file)


def load_most_recent_results(results_folder, path):
    most_recent_experiment_path = sorted(os.listdir(results_folder))[-1]
    path = f"{most_recent_experiment_path}/{path}"
    file_path = os.path.join(results_folder, path)
    with open(file_path, 'rb') as file:
        data = pk.load(file)
    return data

def load_most_recent_model(path):
    recent_run_path = sorted(os.listdir(path))[-1]

    model_path = os.path.join(path, recent_run_path, '0', 'model.pth')
    config_file_path = os.path.join(path, recent_run_path, 'config.pk')
    with open(config_file_path, 'rb') as file:
        config_dict = pk.load(file)
    model = build_model(config_dict)
    model.load_state_dict(torch.load(model_path))
    return model, config_dict

def load_results(results_folder, path):
    file_path = f"{results_folder}/{path}"
    with open(file_path, 'rb') as file:
        data = pk.load(file)
    return data

def load_model(results_folder):
    model_path = os.path.join(results_folder, '0', 'model.pth')
    config_file_path = os.path.join(results_folder, 'config.pk')
    with open(config_file_path, 'rb') as file:
        config_dict = pk.load(file)
    model = build_model(config_dict)
    model.load_state_dict(torch.load(model_path))
    return model, config_dict


def get_abs_path_to_src():
    cwd = os.getcwd()
    src_abs_path_index = cwd.split("/").index("src")
    return "/".join(os.getcwd().split("/")[:src_abs_path_index + 1])

def get_abs_path_to_results_folder():
    cwd = os.getcwd()
    project_root_abs_path_index = cwd.split("/").index("relu_code_pgm")
    path_to_root = "/".join(os.getcwd().split("/")[:project_root_abs_path_index + 1])
    return f"{path_to_root}/{RESULTS_FOLDER}"
def get_abs_path_from_src(paths_strings):
    subpath = "/".join(paths_strings)
    src_abs_path = get_abs_path_to_src()
    return f'{src_abs_path}/{subpath}'
