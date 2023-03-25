from dataclasses import dataclass, field
import torch
from src.config import get_config
import numpy as np
from file_utils import store_results
flatten_list = lambda irregular_list:[element for item in irregular_list for element in flatten_list(item)] if type(irregular_list) is list else [irregular_list]

@dataclass
class MNISTResult():
    codes: list[tuple[bool]]
    actual: int
    predicted: int

# Returns a list of MNISTResult
def iterate_and_collect(loader, network):
    network.eval()  # put the model in eval mode
    # collect all activated codes by the data in loader
    results = []
    with torch.no_grad():
        for data, target in loader:
            output, batch_code_tensor = network.forward_get_code(data)
            pred = [int(out.cpu().detach().numpy())
                    for out in output.data.max(1, keepdim=True)[1]]
            if isinstance(target, torch.Tensor):
                target = target.cpu().detach().numpy()
            if isinstance(target, tuple):
                target = np.asarray(target)
            batch_code_tensor = flatten_list(batch_code_tensor)
            batch_code_numpy = [layer.cpu().detach().numpy()
                                for layer in batch_code_tensor]
            for b in range(batch_code_numpy[0].shape[0]): # iterate over each sample of batch
                actual = target[b]
                predicted = pred[b]
                code_chunks_per_layer = [] # contains codes for each layer
                for layer_idx in range(len(batch_code_numpy)):
                    code_chunks_per_layer.append(tuple(batch_code_numpy[layer_idx][b].astype(bool)))
                results.append(MNISTResult(codes= code_chunks_per_layer, actual = actual, predicted = predicted))
    return results


def compile_and_store_mnist_results(network, test_loader, train_loader, run_path, result_prefix):
    # iterate through the test set and check which codes are being activated
    test_results = iterate_and_collect(test_loader, network)
    store_results(f"{result_prefix}test", test_results, run_path)
    train_results = iterate_and_collect(train_loader, network)
    store_results(f"{result_prefix}train", train_results, run_path)