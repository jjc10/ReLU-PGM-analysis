from file_utils import load_most_recent_results, load_most_recent_model
from src.config import RESULTS_FOLDER, get_config
import numpy as np
import torch
import torch.nn.functional as F
from src.data import get_mnist_data

class LinearSoftMax(torch.nn.Module):
    def __init__(self, input_size, weights, bias, output_size = 10):
        super(LinearSoftMax, self).__init__()
        self.input_size = input_size
        self.weights = torch.from_numpy(weights)
        self.bias = torch.from_numpy(bias)
        self.linear = torch.nn.Linear(self.input_size, output_size)
        # nn.Parameter(F.softmax(self.layer_weights,dim=0))
        with torch.no_grad():
            self.linear.weight = torch.nn.Parameter(self.weights)
            self.linear.bias = torch.nn.Parameter(self.bias)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.linear(x)
        log_out = F.log_softmax(x, dim=1)
        return log_out

# Accepts a relu activation code and generates a linear softmax model from it.
def NN_to_logreg(model, list_codes_per_layer):
    model.eval()
    weights = []
    biases = []
    for i, layer_code in enumerate(list_codes_per_layer):
        # build matrix to cancel off ReLU's
        layer = model.first_layer if i == 0 else model.list_layers[i - 1] # fix this so we can access the layers directly in model.list_layer
        cancel_matrix = np.eye(layer.weight.shape[0])
        for r in range(len(layer_code)):
            cancel_matrix[r, r] = layer_code[r]
        new_weight = np.matmul(cancel_matrix, layer.weight.detach().numpy())
        new_bias = np.matmul(cancel_matrix, layer.bias.detach().numpy())
        weights.append(new_weight)
        biases.append(new_bias)
    # Add last layer fed into softmax
    weights.append(model.last_layer.weight.detach().numpy())
    biases.append(model.last_layer.bias.detach().numpy())

    # Combine all weights and biases into a single
    combined_weight = weights[len(weights) - 1]
    combined_bias = biases[len(biases) - 1]
    for i in range(len(weights) - 2, -1, -1):
        combined_bias = combined_bias + np.matmul(combined_weight, biases[i]) # This line should go before the below one
        combined_weight = np.matmul(combined_weight, weights[i])
    return LinearSoftMax(model.input_size, combined_weight, combined_bias)

class LinearMixture(torch.nn.Module):
    def __init__(self, linearized_models_with_weights, input_size, output_size = 10):
        super(LinearMixture, self).__init__()
        self.linearized_models_with_weights = linearized_models_with_weights
        self.normalized_models = self._normalize_mixture_weights()
        self.input_size = input_size
        self.output_size = output_size

    def _normalize_mixture_weights(self):
        total_weight = 0
        for t in self.linearized_models_with_weights:
            model_weight = t[1]
            total_weight += model_weight
        normalized_models = []
        for t in self.linearized_models_with_weights:
            normalized_models.append((t[0], t[1] / total_weight))
        return normalized_models

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = torch.zeros((x.shape[0], self.output_size)) # batch size x output
        for t in self.normalized_models:
            model_output = t[0](x)
            weighted_output = model_output * t[1]
            out = out + weighted_output
        return out


def tuple_code_to_list(full_code):
    hidden_layer_size = get_config()['hidden_size']
    depth = get_config()['depth']
    list_codes_per_layer=  []
    for i in range(depth):
        start_idx = i * hidden_layer_size
        end_idx = start_idx + hidden_layer_size
        layer_code = full_code[start_idx : end_idx]
        layer_code = [int(c) for c in layer_code]
        list_codes_per_layer.append(list(layer_code))
    return list_codes_per_layer

def create_mixture(model, num_top_codes, code_freqs):
    num_code = 0
    top_code_and_weights = []
    for k in code_freqs.keys():
        split_per_layer = tuple_code_to_list(k)
        top_code_and_weights.append((split_per_layer, code_freqs[k]))
        num_code += 1
        if num_code == num_top_codes:
            break
    mixture = []
    for t in top_code_and_weights:
        # create model from each code and add it to the mixture
        mixture.append((NN_to_logreg(model, t[0]), t[1]))
    mixture = LinearMixture(mixture, model.input_size, 10)
    return mixture

def compute_accuracy_of_linearized_model(linear_classifier, model):
    train_loader, test_loader = get_mnist_data(get_config())
    correct_preds_count = 0
    for X, y in test_loader:
        X_double = X.double()
        output = linear_classifier(X_double)
        model_output, codes = model.forward_get_code(X)
        layer_1 = codes[0].detach().numpy().astype(int)
        layer_2 = codes[1].detach().numpy().astype(int)
        zipped = list(zip(layer_1, layer_2))
        combined_codes = [tuple(code[0]) + tuple(code[1]) for code in zipped]
        # indices_for_matching_code = [i for i in range(len(combined_codes)) if combined_codes[i] == top_codes[0]]
        pred = [int(out.cpu().detach().numpy())
                for out in output.data.max(1, keepdim=True)[1]]
        model_pred = [int(out.cpu().detach().numpy())
                      for out in model_output.data.max(1, keepdim=True)[1]]
        correct_preds = (pred == y.detach().numpy())
        correct_preds_count += correct_preds.sum()


def compute_accuracy_of_mixture(mixture):
    train_loader, test_loader = get_mnist_data(get_config())
    correct_preds_count = 0
    for X, y in test_loader:
        X_double = X.double()
        output = mixture(X_double)

        pred = [int(out.cpu().detach().numpy())
                for out in output.data.max(1, keepdim=True)[1]]
        correct_preds = (pred == y.detach().numpy())
        correct_preds_count += correct_preds.sum()
    accuracy = 100. * correct_preds_count / len(test_loader.dataset)
    print(f"Accuracy of mixture is {accuracy}")
