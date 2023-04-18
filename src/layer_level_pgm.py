from src.config import RESULTS_FOLDER
from file_utils import load_most_recent_results, load_most_recent_model
import numpy as np
import networkx as nx
from utils import serialize_networkx_graph, deserialize_networkx_graph
from mnist.model_linearization import build_prob_tree_for_mixture, normalize_prob_tree, extract_weighted_linear_models_from_prob_tree, create_conditional_mixture, compute_accuracy_of_mixture

NODE = 'node'
post_train_run_0 = load_most_recent_results(f'../{RESULTS_FOLDER}', "0/post_train.pk")
model = load_most_recent_model(f'../{RESULTS_FOLDER}')
model.eval()
mnist_code_results = post_train_run_0
class LayerGraphNode:
    def __init__(self, layer_number, model, parent_node = None):
        assert ((layer_number == 0 and not parent_node) or (layer_number > 0 and parent_node))
        self.layer_number = layer_number
        self.model = model
        self.parent_node = parent_node
        self.number_of_parents = 1 if self.parent_node else 0
        self.number_of_neurons = self._get_number_of_neurons()
        self._setup_cpt_tensor()

    def valid_rv_values(self):
        return list(range(0, 2**self.number_of_neurons))

    def _setup_cpt_tensor(self):
        if self.number_of_parents == 0: # 1D tensor
            self.cpt = np.zeros((len(self.valid_rv_values())))
        else: # 2D tensor since we have a parent
            self.cpt = np.zeros((len(self.valid_rv_values()), len(self.parent_node.valid_rv_values())))

    def _get_number_of_neurons(self):
        # Fix model so we don't have to do this shit
        if self.layer_number == 0: # 0th hidden layer.
            return self.model.first_layer.out_features
        else:
            return self.model.list_layers[self.layer_number - 1].out_features # add 1 since 0th layer is the first layer

class TargetGraphNode:
    def __init__(self, model, parent_node):
        self.number_of_targets = model.last_layer.out_features # Right before softmax
        self.parent_node = parent_node
        self.number_of_neurons_in_parent = parent_node.number_of_neurons
        self._setup_cpt_tensor()

    def _setup_cpt_tensor(self):
        self.cpt = np.zeros((self.number_of_targets, 2 ** self.number_of_neurons_in_parent))

def create_layer_level_pgm_structure_from_model(model):
    depth = model.depth
    prev_node = None

    BN = nx.DiGraph()
    for layer_idx in range(depth):
        node = LayerGraphNode(layer_idx, model, prev_node)
        BN.add_node(layer_idx, node=node)
        if prev_node is not None:
            BN.add_edge(prev_node.layer_number, node.layer_number)
        prev_node = node
    # Target node
    target_node = TargetGraphNode(model, prev_node)
    BN.add_node('T', node=target_node)
    BN.add_edge(prev_node.layer_number, 'T')
    return BN

def convert_boolean_code_to_int(boolean_code):
    binary_string = np.array2string(np.array(boolean_code, dtype=int), separator='')[1:-1] # remove [ and ] at each end.
    return int(binary_string, 2)

def update_layer_level_cpd_from_code_freq(bayes_net, mnist_result_codes):
    # first perform counts, we normalize later in reverse topological order
    for mnist_result_code in mnist_result_codes:
        parent_value = None
        # Deal with hidden layers
        for layer_idx, layer_code in enumerate(mnist_result_code.codes): # loop per layer
            if parent_value is None:
                rv_value = convert_boolean_code_to_int(layer_code)
                bayes_net.nodes[layer_idx][NODE].cpt[rv_value] += 1
            else:
                rv_value = convert_boolean_code_to_int(layer_code)
                bayes_net.nodes[layer_idx][NODE].cpt[rv_value, parent_value] += 1
            parent_value = rv_value
        # Deal with last layer
        last_layer_rv_value = convert_boolean_code_to_int(mnist_result_code.codes[-1])
        predicted = mnist_result_code.predicted
        bayes_net.nodes['T'][NODE].cpt[predicted, last_layer_rv_value] += 1
    reverse_topological_order = list(bayes_net.nodes)[::-1]
    for node in reverse_topological_order: # Loop in reverse order because we need to keep the parent's counts for the children so normalize children first
        graph_node = bayes_net.nodes[node][NODE]
        parent_node = graph_node.parent_node
        if not parent_node:
            total_sum = np.sum(graph_node.cpt)
            graph_node.cpt = graph_node.cpt / total_sum
        else:
            parent_cpt = parent_node.cpt
            # a cpt is (RV, PARENT_RV) so we sum over the parent rv's to count all values. Similar to a marginalization
            parent_values_count = np.sum(parent_cpt, axis = 1) if parent_cpt.ndim > 1 else parent_cpt
            graph_node.cpt = np.nan_to_num(graph_node.cpt / parent_values_count)


def create_layer_level_pgm_from_relu_codes(model, mnist_code_results):
    bn = create_layer_level_pgm_structure_from_model(model)
    update_layer_level_cpd_from_code_freq(bn, mnist_code_results)
    return bn

bn = deserialize_networkx_graph("../networkx_storage", "layer_level_pgm")

def hyperparameter_tune_conditional_mixture(model, bn, number_of_codes_per_layer_list,):
    for number_of_codes_per_layer in number_of_codes_per_layer_list:
        prob_tree = build_prob_tree_for_mixture(number_of_codes_per_layer, bn)
        normalize_prob_tree(prob_tree)
        weighted_linear_models = extract_weighted_linear_models_from_prob_tree(prob_tree, [8, 8])
        conditional_mixture = create_conditional_mixture(model, weighted_linear_models)

        acc = compute_accuracy_of_mixture(conditional_mixture)
        print(f"Accuracy of conditional mixture for {number_of_codes_per_layer} is {acc}")

hyperparams = [[2,2], [2, 3], [2, 8], [4, 3], [4, 5], [4, 8], [5, 3], [5, 5], [5, 8], [6, 4], [6, 8], [10, 5], [10, 10]]
hyperparameter_tune_conditional_mixture(model, bn, hyperparams)
