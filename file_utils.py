
import os
import time
import pickle as pk


def generate_run_id():
    now = int(time.time())
    string_version_now = str(now)
    return string_version_now


def store_results(file_name, data_to_store, run_path):
    file_path = os.path.join(run_path, file_name+'.pk')
    with open(file_path, 'wb') as file:
        pk.dump(data_to_store, file)


def load_most_recent_results(path):
    recent_run_path = sorted(os.listdir(path))[-1]
    file_path = os.path.join(path, recent_run_path, 'results.pk')
    with open(file_path, 'rb') as file:
        data = pk.load(file)
    return data
