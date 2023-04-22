import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score
import pandas as pd
from src.utils import get_prob_dictionary_from_activation_results
from src.mnist.mnist_experiment import ReluActivationResult

NODE = 'node'
def look_at_point(data, targets):

    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.tight_layout()
        plt.imshow(data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()
    plt.close()


def invert_dict(in_dict):
    out_dict = {}
    for _, value in in_dict.items():
        if value not in out_dict:
            out_dict[value] = 1
        else:
            out_dict[value] += 1
    return out_dict


def plot_code_histograms(codes_histograms, labels, prefix_title, prefix_file):
    binary_counter = {0: 0, 1: 0}

    for i, codes_histogram in enumerate(codes_histograms):
        # count the fraction of 1 and 0
        for key, value in codes_histogram.items():
            for c in key:
                if c in binary_counter:
                    binary_counter[c] += value
        values = list(codes_histogram.values())
        values.sort()
        num_samples = np.sum(values)
        values = [v/num_samples for v in values]
        plt.bar(range(len(values)), values, 1,
                label=labels[i] + ', weighted 1/0 ratio {:2.2f}'.format(binary_counter[1]/binary_counter[0]))
    plt.legend()
    plt.xlabel('codes')
    plt.xlabel('frequency')
    plt.title(prefix_title+' code histograms')
    plt.savefig('figures/'+prefix_file + 'code_histogram.pdf')
    plt.close()
    # generate side plot
    for i, codes_histogram in enumerate(codes_histograms):
        count_histogam = invert_dict(codes_histogram)
        values = list(count_histogam.values())
        values.sort()
        plt.bar(range(len(values)), values, 1, label=labels[i])
    plt.legend()
    plt.xlabel('count')
    plt.xlabel('frequency of codes')
    plt.title(prefix_title+' count histograms')
    plt.savefig('figures/'+prefix_file + 'count_histogram.pdf')
    plt.close()


def get_code_class_hist(result_dict, min_code_occupancy=1, max_code_occupancy=10000, type='true'):
    
        
    key_set = result_dict.keys()
    counts = [0 for i in range(10)]
    accuracies = [[] for i in range(10)]
    for key in key_set:
        if type == 'true':
            tuple_list = [t[0] for t in result_dict[key]]
        elif type == 'predicted':
            tuple_list = [t[1] for t in result_dict[key]]
        
        
        a = accuracy_score([t[0] for t in result_dict[key]], [t[1] for t in result_dict[key]])
        if len(tuple_list) >= min_code_occupancy and len(tuple_list) < max_code_occupancy:
            nunique_codes = len(np.unique(tuple_list))
            counts[nunique_codes-1] += 1
            accuracies[nunique_codes-1].append(a)
    acc = [100*np.mean(a) for a in accuracies]
    return counts, acc


def plot_code_class_density(result_dict, prefix_title, prefix_file):
    def plot(type):
        if type == 'true':
            title = 'true_classes_'
        else:
            title = 'predicted_'
        populations = []
        a = []
        sizes = [1, 2, 5, 10, 50]
        for i in range(1, len(sizes)):
            pop, accs = get_code_class_hist(
                result_dict, min_code_occupancy=sizes[i-1], max_code_occupancy=sizes[i], type=type)
            populations.append(pop)
            a.append(accs)
        pop, accs = get_code_class_hist(
            result_dict, min_code_occupancy=50, type=type)
        a.append(accs)
        print(a)
        populations.append(pop)
        populations = np.array(populations)
        x = np.arange(10) + 1
        plt.bar(x, populations[0])
        plt.bar(x, populations[1], bottom=populations[0])
        plt.bar(x, populations[2], bottom=populations[0]+populations[1])
        plt.bar(x, populations[3], bottom=populations[0] +
                populations[1]+populations[2])
        plt.bar(x, populations[4], bottom=populations[0] +
                populations[1]+populations[2]+populations[3])
        plt.xlabel("Number of unique true classes per code")
        plt.ylabel("Frequency")
        plt.title(prefix_title+'Num. of codes ')
        plt.legend(["10", "11-20", "21-30", "31-40", "50+"],
                   title="Num. of instances in code:")
        plt.xticks(np.arange(0, 11))
        plt.savefig('figures/' + prefix_file + title+"_per_code.pdf")
        plt.close()
    plot(type='true')
    plot(type='predicted')

def plot_distribution_of_hidden_codes_first_layer(bn):
    dist = bn.nodes[0][NODE].cpt
    ordered_asc = np.argsort(dist)
    ordered_desc = ordered_asc[::-1]
    plt.plot(bn.nodes[0][NODE].cpt[ordered_desc])
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    plt.xlabel("Activation code on first hidden layer (hidden for clarity)")
    plt.ylabel("Frequency of code")
    plt.legend()
    plt.savefig("../figures/layer_0_code_freq_no_ticks.pdf")

#TODO: Adapt this to work any layer. The problem is that for layers deeper, the parent marginal dist is not readily available
# We need to marginalize the parent's probability before applying the law of total prob.
def plot_cum_prob_of_top_codes_l1_conditioned_l0(bn, level, num_of_top_codes_previous_layer, num_of_top_codes_in_current_layer):
    assert level > 0
    graph_node = bn.nodes[level][NODE]
    parent_node = graph_node.parent_node
    dist_parent = np.sum(parent_node.cpt, axis = 1) if parent_node.cpt.ndim > 1 else parent_node.cpt # Get the dist
    ordered_l0_asc = np.argsort(dist_parent)
    ordered_l0_desc = ordered_l0_asc[::-1]
    cum_sum = []
    for code in ordered_l0_desc[:num_of_top_codes_previous_layer]:
        conditional_prob = graph_node.cpt[:, code]
        top_h1_codes_indices = np.argpartition(conditional_prob, -num_of_top_codes_in_current_layer)[-num_of_top_codes_in_current_layer:]
        cum_prob_sum = np.sum(conditional_prob[top_h1_codes_indices])
        cum_sum.append(cum_prob_sum)
    plt.rcParams["axes.titlesize"] = 8
    plt.bar(x=range(0, len(cum_sum)), height=cum_sum)
    plt.xticks(ticks=range(0, len(cum_sum)), labels=ordered_l0_desc[:num_of_top_codes_previous_layer])
    plt.xlabel("Popular code in H0")
    plt.ylabel("Cumulative conditional probability of H1 given code in H0")
    # plt.title(f"Cumulative conditional probability of top {num_of_top_codes_in_current_layer} codes in L1 conditioned on top codes in L0\n ")
    # plt.show()
    plt.savefig("../figures/cumulative_prob_top1_given_top0.pdf")

def heatmap_target_given_last(bn):
    non_ordered_df = pd.DataFrame(bn.nodes['T'][NODE].cpt)
    fig, ax = plt.subplots(figsize=(120,10))
    sns.heatmap(non_ordered_df, annot=False,cmap='Reds', fmt='g', linewidths=0.5, ax=ax)
    plt.rcParams["axes.titlesize"] = 18
    plt.xlabel("Code in last hidden layer")
    plt.ylabel("Predicted target classes")
    # plt.title("Heatmap showing likely target class given code in last hidden layer")
    plt.legend()
    plt.savefig("../figures/heatmap_target_given_l1.pdf", bbox_inches='tight')


def plot_full_code_distributions(relu_activation_results: list[ReluActivationResult]):
    count_dict = get_prob_dictionary_from_activation_results(relu_activation_results)
    probabilities = np.array(list(count_dict.values()))
    sorted_indices_asc = np.argsort(probabilities)
    sorted_indices_desc = sorted_indices_asc[::-1]
    plt.plot(probabilities[sorted_indices_desc])
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    plt.xlabel("Activation codes over full network (hidden for clarity)")
    plt.ylabel("Frequency of code")
    plt.legend()
    plt.savefig("../figures/full_code_freq_mnist_3_12.pdf")


