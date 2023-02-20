import os
from file_utils import generate_run_id, store_results
from src.analysis import compile_results, compile_imagenet_results
from src.train import train_model
from src.model import build_model
from src.config import FIGURE_FOLDER, RESULTS_FOLDER, get_config, set_randomness, set_up_paths, get_pretrained_imagenet_config
from src.data import get_mnist_data
from src.data import get_imagenet_test_data
from src.resnet.resnet import resnet18
import numpy as np
import torch.utils.data as data_utils
import torch
from torchvision import transforms
from PIL import Image

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # there are a few greyscale images
    transforms.RandomResizedCrop((500, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
prison = Image.open("./imagenet/imagenet-sample-images-master/n04005630_prison.JPEG")
prison_t = transform(prison)
shark = Image.open("./imagenet/n01484850_great_white_shark.jpeg")
shark_t = transform(prison)
shark_t = prison_t[None, :]
def run_mnist_experiment():
    set_up_paths([FIGURE_FOLDER, RESULTS_FOLDER])
    config_dict = get_config()
    set_randomness(seed=3)
    train_loader, test_loader = get_mnist_data(config_dict)
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
        network = build_model(input_size, config_dict)
        # look at relu activated in the network by the data before training
        init_compiled_results = compile_results(network, test_loader, train_loader, result_prefix='init_')
        train_model(network, train_loader, test_loader, config_dict, run_path_trial)
        # look at relu activated in the network by the data after training
        post_compiled_results = compile_results(network, test_loader, train_loader, result_prefix='post_')
        result_dict[trial] = post_compiled_results
        result_dict[trial].update(init_compiled_results)
    store_results('results', result_dict, run_path)
    store_results('config', config_dict, run_path)

def run_imagenet_experiment(subset = None):
    set_up_paths([FIGURE_FOLDER, RESULTS_FOLDER])
    config_dict = get_pretrained_imagenet_config()
    set_randomness(seed=3)
    test_loader = get_imagenet_test_data(config_dict, './imagenet/imagenet-sample-images-master', subset = subset)
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    input_size = np.prod(example_data[0].shape)


    run_id = generate_run_id()
    run_path = os.path.join(RESULTS_FOLDER, run_id)
    set_up_paths([run_path])

    result_dict = {}
    for trial in range(config_dict['trials']):
        run_path_trial = os.path.join(run_path, str(trial))
        set_up_paths([run_path_trial])
        network = resnet18(input_size, pretrained=True)
        # look at relu activated in the network by the data before training
        compiled_results = compile_imagenet_results(network, test_loader, result_prefix='init_')
        result_dict[trial] = compiled_results
    store_results('results', result_dict, run_path)
    store_results('config', config_dict, run_path)

run_imagenet_experiment(1)