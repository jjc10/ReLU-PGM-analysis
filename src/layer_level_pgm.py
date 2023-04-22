import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from mnist.model_linearization import build_prob_tree_for_mixture, \
    normalize_prob_tree, extract_weighted_linear_models_from_prob_tree, create_conditional_mixture, \
    compute_accuracy_of_mixture, LinearModelMode
from src.file_utils import load_most_recent_results, load_most_recent_model, get_abs_path_to_results_folder, load_results, load_model
from utils import deserialize_networkx_graph, serialize_networkx_graph
from src.plot_util import heatmap_target_given_last, plot_cum_prob_of_top_codes_l1_conditioned_l0,\
    plot_distribution_of_hidden_codes_first_layer, plot_full_code_distributions
NODE = 'node'

results_folder = get_abs_path_to_results_folder()
# post_train_run_0 = load_most_recent_results(results_folder, "0/post_train.pk")

post_train_run_0 = load_results(f'{results_folder}/cifar_2_8_1682107640', "0/post_train.pk")
# model, config_dict = load_most_recent_model(results_folder)
model, config_dict = load_model(f'{results_folder}/cifar_2_8_1682107640')

model.eval()
mnist_code_results = post_train_run_0


class LayerGraphNode:
    def __init__(self, layer_number, model, parent_node=None):
        assert ((layer_number == 0 and not parent_node) or (layer_number > 0 and parent_node))
        self.layer_number = layer_number
        self.model = model
        self.parent_node = parent_node
        self.number_of_parents = 1 if self.parent_node else 0
        self.number_of_neurons = self._get_number_of_neurons()
        self._setup_cpt_tensor()

    def valid_rv_values(self):
        return list(range(0, 2 ** self.number_of_neurons))

    def _setup_cpt_tensor(self):
        if self.number_of_parents == 0:  # 1D tensor
            self.cpt = np.zeros((len(self.valid_rv_values())))
        else:  # 2D tensor since we have a parent
            # Current variable is first dimension, parents are subsequent dimensions meaning a column fixes the value of the parent,
            # giving us the full distribution of the current RV conditioned on its parent.
            self.cpt = np.zeros((len(self.valid_rv_values()), len(self.parent_node.valid_rv_values())))

    def _get_number_of_neurons(self):
        # Fix model so we don't have to do this shit
        if self.layer_number == 0:  # 0th hidden layer.
            return self.model.first_layer.out_features
        else:
            return self.model.list_layers[
                self.layer_number - 1].out_features  # add 1 since 0th layer is the first layer


class TargetGraphNode:
    def __init__(self, model, parent_node):
        self.number_of_targets = model.last_layer.out_features  # Right before softmax
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
    binary_string = np.array2string(np.array(boolean_code, dtype=int), separator='')[
                    1:-1]  # remove [ and ] at each end.
    return int(binary_string, 2)


def update_layer_level_cpd_from_code_freq(bayes_net, mnist_result_codes):
    # first perform counts, we normalize later in reverse topological order
    for mnist_result_code in mnist_result_codes:
        parent_value = None
        # Deal with hidden layers
        for layer_idx, layer_code in enumerate(mnist_result_code.codes):  # loop per layer
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
    for node in reverse_topological_order:  # Loop in reverse order because we need to keep the parent's counts for the children so normalize children first
        graph_node = bayes_net.nodes[node][NODE]
        parent_node = graph_node.parent_node
        if not parent_node:
            total_sum = np.sum(graph_node.cpt)
            graph_node.cpt = graph_node.cpt / total_sum
        else:
            parent_cpt = parent_node.cpt
            # a cpt is (RV, PARENT_RV) so we sum over the parent rv's to count all values. Similar to a marginalization
            parent_values_count = np.sum(parent_cpt, axis=1) if parent_cpt.ndim > 1 else parent_cpt
            graph_node.cpt = np.nan_to_num(graph_node.cpt / parent_values_count)


def create_layer_level_pgm_from_relu_codes(model, mnist_code_results):
    bn = create_layer_level_pgm_structure_from_model(model)
    update_layer_level_cpd_from_code_freq(bn, mnist_code_results)
    return bn


def hyperparameter_tune_conditional_mixture(model, bn, number_of_codes_per_layer_list):
    acc_mix_train = []
    acc_mix_test = []
    acc_argmax_train = []
    acc_argmax_test = []
    acc_w_argmax_test = []
    acc_w_argmax_train = []
    for number_of_codes_per_layer in number_of_codes_per_layer_list:
        prob_tree = build_prob_tree_for_mixture(number_of_codes_per_layer, bn)
        normalize_prob_tree(prob_tree)
        widths = model.get_layer_widths()
        weighted_linear_models = extract_weighted_linear_models_from_prob_tree(prob_tree, widths)
        conditional_mixture = create_conditional_mixture(model, weighted_linear_models)

        acc_mix_test.append(compute_accuracy_of_mixture(conditional_mixture, LinearModelMode.MIXTURE)[0])
        acc_argmax_test.append(compute_accuracy_of_mixture(conditional_mixture, LinearModelMode.ARGMAX)[0])
        acc_w_argmax_test.append(compute_accuracy_of_mixture(conditional_mixture, LinearModelMode.WEIGHTED_ARGMAX)[0])
    plt.figure(figsize=(10, 8), dpi=80)
    plt.plot(acc_mix_test, label='Mixture test')
    plt.plot(acc_argmax_test, label='Argmax test')
    # plt.plot(acc_argmax_train, label='Argmax train')
    plt.plot(acc_w_argmax_test, label='Weighted_Argmax test')
    # plt.plot(acc_w_argmax_train, label='Weighted_Argmax train')
    labels = [str(hyperparam) for hyperparam in number_of_codes_per_layer_list]
    plt.xticks(np.arange(len(acc_mix_test)), labels, rotation='vertical')
    # plt.plot(acc_entropy_test, label='Entropy, test')
    # plt.plot(acc_entropy_train, label='Entropy, train')
    plt.xlabel("Number of top codes considered per layer conditioned on previous layer")
    plt.ylabel("Percentage accuracy on test set")
    plt.legend()
    plt.savefig(f"../figures/conditional_mixture_d{len(number_of_codes_per_layer_list[0])}_w{model.get_layer_widths()[0]}.pdf")


depth = config_dict['depth']
hidden_size = config_dict['hidden_size']
bn = create_layer_level_pgm_from_relu_codes(model, mnist_code_results)
serialize_networkx_graph(bn, "../networkx_storage", f"layer_level_pgm_cifar_{depth}_{hidden_size}")

# bn = deserialize_networkx_graph("../networkx_storage", f"layer_level_pgm_mnist_{depth}_{hidden_size}")

# hyperparams = [[1, 1, 1], [1, 2, 2], [2,2,2], [2, 3, 3], [2, 8, 8], [3, 2, 1], [3, 1, 1], [3, 2, 1], [3, 2, 2], [3, 3, 2], [3, 4, 4], [3, 5, 5], [4, 4, 3], [4, 5, 5], [4, 8, 8], [5, 3, 3], [5, 5, 5], [10, 2, 1], [10, 5, 5]]
# hyperparams = [[1, 2, 2, 2], [2, 1, 1, 1], [2, 2, 2, 2], [2, 2, 1, 1], [3, 2, 1, 1], [3, 3, 1, 1], [4, 2, 1, 1]]
# hyperparams = [[4, 4, 2, 2],[6, 2, 1, 1], [5, 2, 1, 1], [6, 3, 2, 1], [10, 2, 2, 1]]
hyperparams = [[1, 1], [1, 2], [2, 2], [2, 4], [2, 6], [3, 2], [3, 5], [4, 3], [4, 5], [5, 2], [5, 8], [8, 4], [8, 8], [10, 3]]
hyperparameter_tune_conditional_mixture(model, bn, hyperparams)


# plot_distribution_of_hidden_codes_first_layer(bn)
# plot_cum_prob_of_top_codes_l1_conditioned_l0(bn, 1, 10, 10)
# heatmap_target_given_last(bn)
# plot_full_code_distributions(mnist_code_results)