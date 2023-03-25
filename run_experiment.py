import os
from file_utils import generate_run_id, store_results
from src.mnist.mnist_experiment import compile_and_store_mnist_results
from src.train import train_model
from src.model import build_model
from src.config import FIGURE_FOLDER, RESULTS_FOLDER, get_config, set_randomness, set_up_paths, get_pretrained_imagenet_config
from src.data import get_mnist_data
import numpy as np
from torchvision import transforms



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # there are a few greyscale images
    transforms.RandomResizedCrop((500, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

def run_mnist_experiment():
    set_up_paths([FIGURE_FOLDER, RESULTS_FOLDER])
    config_dict = get_config()
    set_randomness(seed=3)
    train_loader, test_loader = get_mnist_data(config_dict)
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)


    # get dimension of flatten input 1x28x28 -> 784
    input_size = np.prod(example_data[0].shape)
    config_dict['input_size'] = input_size # keep this in config in case

    run_id = generate_run_id()
    run_path = os.path.join(RESULTS_FOLDER, run_id)
    set_up_paths([run_path])
    for trial in range(config_dict['trials']):
        run_path_trial = os.path.join(run_path, str(trial))
        set_up_paths([run_path_trial])
        network = build_model(config_dict)
        # look at relu activated in the network by the data before training
        compile_and_store_mnist_results(network, test_loader, train_loader, run_path_trial, result_prefix='init_')
        train_model(network, train_loader, test_loader, config_dict, run_path_trial)
        # look at relu activated in the network by the data after training
        compile_and_store_mnist_results(network, test_loader, train_loader, run_path_trial, result_prefix='post_')
        store_results('config', config_dict, run_path)

run_mnist_experiment()

