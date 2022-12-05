from src.analysis import check_results, compile_results
from src.train import train_model
from src.model import build_model
from src.config import get_config, set_randomness
from src.data import get_data
from src.plot_util import look_at_point
import numpy as np
# Set config and load data
config_dict = get_config()
set_randomness()
train_loader, test_loader = get_data(config_dict)
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
# look_at_point(example_data, example_targets)

# get dimension of flatten input 1x28x28 -> 784
input_size = np.prod(example_data[0].shape)

network = build_model(input_size, config_dict)

# look at relu activated in the network by the data before training
init_compiled_results = compile_results(network, test_loader, train_loader)

train_model(network, train_loader, test_loader, config_dict)

# look at relu activated in the network by the data after training
post_compiled_results = compile_results(network, test_loader, train_loader)

check_results(init_compiled_results, prefix='init_')
check_results(post_compiled_results, prefix='post_')
