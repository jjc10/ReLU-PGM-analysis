import os
from file_utils import generate_run_id, store_results
from src.analysis import compile_results
from src.train import train_model
from src.model import build_model
from src.config import FIGURE_FOLDER, RESULTS_FOLDER, get_config, set_randomness, set_up_paths
from src.data import get_data
from src.resnet.resnet import resnet18
import numpy as np
# Set up path, set config and load data

set_up_paths([FIGURE_FOLDER, RESULTS_FOLDER])
config_dict = get_config()
set_randomness(seed=3)
train_loader, test_loader = get_data(config_dict)
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


# get dimension of flatten input 1x28x28 -> 784
input_size = np.prod(example_data[0].shape)


run_id = generate_run_id()
run_path = os.path.join(RESULTS_FOLDER, run_id)
set_up_paths([run_path])

result_dict = {}
for trial in range(config_dict['trials']):
    run_path_trial = os.path.join(run_path, str(trial))
    set_up_paths([run_path_trial])
    # network = build_model(input_size, config_dict)
    network = resnet18(num_classes = 10)
    # look at relu activated in the network by the data before training
    init_compiled_results = compile_results(network, test_loader, train_loader, result_prefix='init_')
    train_model(network, train_loader, test_loader, config_dict, run_path_trial)
    # look at relu activated in the network by the data after training
    post_compiled_results = compile_results(network, test_loader, train_loader, result_prefix='post_')
    result_dict[trial] = post_compiled_results
    result_dict[trial].update(init_compiled_results)
store_results('results', result_dict, run_path)
store_results('config', config_dict, run_path)
