
import os
import time
import pickle as pk
import torch
from src.model import build_model


def generate_run_id():
    now = int(time.time())
    string_version_now = str(now)
    return string_version_now


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
    return model
