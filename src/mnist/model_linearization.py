from file_utils import load_most_recent_results, load_most_recent_model
from src.config import RESULTS_FOLDER, get_config
import numpy as np
import torch
import torch.nn.functional as F
from src.data import get_mnist_data
import networkx as nx
from queue import Queue
from src.utils import convert_int_to_boolean_code

NODE = 'node'

class LinearSoftMax(torch.nn.Module):
    """
    A simple linear softmax classifier initialized with weights. Corresponds to one single linearized instance of the network
    """
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
        cancel_matrix = np.eye(layer.weight.shape[0]) # layer.weight is WxWp where W is the width of the layer and Wp is the width of the previous layer.
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

def create_conditional_mixture(model, weighted_codes):
    mixture = []
    for t in weighted_codes:
        # create model from each code and add it to the mixture
        mixture.append((NN_to_logreg(model, t[0]), t[1]))
    mixture = LinearMixture(mixture, model.input_size, 10)
    return mixture

def get_node_key(node_val, ancestor_key):
    return f'{ancestor_key}-{node_val}'

def get_node_value_from_key(node_key):
    return int(node_key.split("-")[-1])

def build_prob_tree_for_mixture(list_num_top_codes, bayes_net):
    """Builds a "prob tree" from a Bayes net containing layer-wise activation probabilities given a list of top codes to consider
    per layer.
    A "prob tree" is simply a tree data structure to easily hold the information needed to build linearized models as conditional mixtures.
    Each node has a key and a probability. It starts with a root node ('R', 1) whose n0 children represent the n0 most common code
    for hidden layer 0. Each of these nodes encodes the decimal value of the binary relu code and its prob. The key is built by appending the integer
    value to the key of the ancestor (this is to guarantee uniqueness and also means that the key encodes the full path easily): ex 'R-157-235'
    Each of these nodes then has children representing the most common n1 codes conditioned on the previous layer.
    The prob tree can then easily be used for normalization since all layers just need to have probs summing to 1.
    It can also easily be used to get the linearized models by simply decoding the path from the leaf keys. The probability of each path
    is then the product of all probabilities on the path to the root.

    :param:
    list_num_top_codes: list[int] each value represents the number of top codes to consider for the layer corresponding to the index.
    bayes_net: A bayesian network holding the conditional probabilities of activation code per layer.

    :returns:
    prob tree matching the list_num_top_codes: we have a single root, the number of nodes at depth k corresponds to list_num_top_codes[k - 1]
    """
    prob_tree = nx.DiGraph()
    prob_tree.add_node('R', prob=1) # root node does not hold any actual info

    # First deal with the 0th layer where there is no conditioning
    ordered_hidden_codes = np.argsort(bayes_net.nodes[0][NODE].cpt)
    ordered_hidden_codes = ordered_hidden_codes[:: -1] # reverse order to be descending
    top_codes = ordered_hidden_codes[:list_num_top_codes[0]]
    for top_code in top_codes:
        prob = bayes_net.nodes[0][NODE].cpt[top_code]
        node_key = get_node_key(top_code, 'R')
        prob_tree.add_node(node_key, prob=prob)
        prob_tree.add_edge('R', node_key)

    # hidden layers with conditioning on previous layer.
    for layer_idx in range(1, len(list_num_top_codes)):
        previous_top_nodes = get_leaves(prob_tree)
        for top_node in previous_top_nodes: # one code at a time
            # get most popular codes conditionally.
            code = get_node_value_from_key(top_node)
            conditional_prob = bayes_net.nodes[layer_idx][NODE].cpt[:, code]
            ordered_codes_conditionally = np.argsort(conditional_prob)
            ordered_codes_conditionally = ordered_codes_conditionally[::-1] # reverse order
            top_codes_conditionally = ordered_codes_conditionally[:list_num_top_codes[layer_idx]]
            for top_code in top_codes_conditionally:
                prob = conditional_prob[top_code]
                node_key = get_node_key(top_code, top_node)
                prob_tree.add_node(node_key, prob=prob)
                prob_tree.add_edge(top_node, node_key)
    return prob_tree

def normalize_prob_tree(prob_tree):
    """
    Normalizes the probabilities of a ProbTree by updating the probabilities to sum to 1 at each layer.
    NOTE: This is done in place, the prob tree will be modified.
    :param prob_tree: prob tree to normalize
    :return:
    """
    root = list(nx.topological_sort(prob_tree))[0]
    q = Queue()
    q.put(root)
    while not q.empty():
        node = q.get()
        children = list(prob_tree.successors(node))
        children_weights = [prob_tree.nodes[n]['prob'] for n in children]
        total_weight = np.sum(children_weights)
        for child in children:
            prob_tree.nodes[child]['prob'] = prob_tree.nodes[child]['prob'] / total_weight
            descendants_of_child = list(prob_tree.successors(child))
            if len(descendants_of_child) > 0:
                q.put(child)

def get_leaves(tree):
    """
    Gets the leaves of a networkx tree
    :param tree:
    :return: list of node keys for all leaves
    """
    return [x for x in tree.nodes() if tree.out_degree(x) == 0 and tree.in_degree(x) == 1]

def get_all_values_to_root(node_key, exclude_root = True):
    """
    Gets all values of the layers activation
    :param node_key: a node key of a prob tree which encodes the full path to the root ('R-154-10') --> 154 --> 10
    :param exclude_root: Drops the root which does not hold any meaningful information since it does not actually represent a layer.
    :return: all ancestors.
    """
    ancestors = list(node_key.split("-"))
    ancestors = ancestors[1:] if exclude_root else ancestors
    ancestors = [int(a) for a in ancestors]
    return ancestors

def get_all_probs_to_root(prob_tree, leaf_key):
    """
    Given a prob tree and a leaf_key, it traverses the tree going upward, grabbing the probability value at each step. When normalized
    this corresponds to the weight of the actual linearized model when computing a mixture.
    :param prob_tree: the normalized prob tree.
    :param leaf_key: key of leaf
    :return: list of all probabilities along the path.
    """
    root_path = list(nx.ancestors(prob_tree, leaf_key))
    root_path.append(leaf_key)
    probs = [prob_tree.nodes[key]['prob'] for key in root_path]
    return probs
def extract_weighted_linear_models_from_prob_tree(prob_tree, widths):
    # start from leaves and go upward
    width = widths[0]
    leaves = get_leaves(prob_tree)
    full_codes_with_weights = []
    for leaf in leaves:
        layer_codes = get_all_values_to_root(leaf)
        binary_codes = [convert_int_to_boolean_code(n, width) for n in layer_codes]
        linear_model_prob = np.prod(get_all_probs_to_root(prob_tree, leaf))
        full_codes_with_weights.append((binary_codes, linear_model_prob))
    return full_codes_with_weights


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
    return accuracy
