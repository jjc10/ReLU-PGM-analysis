from src.table_helper import build_latex_table
import numpy as np
import torch
from src.plot_util import plot_code_histograms, plot_code_class_density
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from src.config import get_config
ACTIVATION_RANGES = [(0, 9.99), (10, 19.99), (20, 29.99), (30, 39.99), (40, 49.99), (50, 59.99), (60, 69.99), (70, 79.99), (80, 89.99), (90, 100)]
def add_code_histo(histo_dict, key, class_per_code_histogram, pred, target):
    if key in histo_dict:
        histo_dict[key] += 1
        class_per_code_histogram[key].append((pred, target))
    else:
        histo_dict[key] = 1
        class_per_code_histogram[key] = [(pred, target)]


flatten_list = lambda irregular_list:[element for item in irregular_list for element in flatten_list(item)] if type(irregular_list) is list else [irregular_list]

def iterate_and_collect(loader, network, result_prefix=''):
    network.eval()  # put the model in eval mode
    code_histogram = {}
    class_per_code_histogram = {}
    code_per_layer_histograms = [{} for _ in range(network.depth)]
    class_per_layer_histograms = [{} for _ in range(network.depth)]
    # collect all activated codes by the data in loader
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
                code_chunks_per_layer = []
                for layer_idx in range(len(batch_code_numpy)):
                    code_chunks_per_layer.append(tuple(batch_code_numpy[layer_idx][b].astype(int)))
                for l, code_chunk in enumerate(code_chunks_per_layer):
                    layer_code_histogram = code_per_layer_histograms[l] # get dictionary for that layer
                    class_per_layer_histogram = class_per_layer_histograms[l]
                    add_code_histo(layer_code_histogram,
                                   code_chunk, class_per_layer_histogram, pred[b], target[b])
                code = [i for sub in code_chunks_per_layer for i in sub]
                code = tuple(code)
                add_code_histo(code_histogram,  code,
                               class_per_code_histogram, pred[b], target[b])

    results = {result_prefix+'code_histogram': code_histogram}
    results[result_prefix+'class_histogram'] = class_per_code_histogram
    for l, code_per_layer_histogram in enumerate(code_per_layer_histograms):
        results[result_prefix+'code_' +
                str(l)+'_histogram'] = code_per_layer_histogram

        results[result_prefix+'class_' +
                str(l)+'_histogram'] = class_per_layer_histograms[l]
    return results


def compile_results(network, test_loader, train_loader, result_prefix):

    # iterate through the test set and check which codes are being activated
    test_results = iterate_and_collect(
        test_loader, network, result_prefix=result_prefix+'test_')
    train_results = iterate_and_collect(
        train_loader, network, result_prefix=result_prefix+'train_')

    percent_test_unseen_train = 0
    for code, val in test_results[result_prefix+'test_code_histogram'].items():
        if code not in train_results[result_prefix+'train_code_histogram']:
            percent_test_unseen_train += val

    compiled_results = test_results
    compiled_results.update(train_results)
    compiled_results[result_prefix+'new_code_test'] = percent_test_unseen_train
    return compiled_results

def compile_imagenet_results(network, test_loader, result_prefix):
    def count_values_in_range(range_tuple, array):
        low = range_tuple[0]
        high = range_tuple[1]
        return ((low <= array) & (array <= high)).sum() / len(array) * 100

    def get_activation_ranges_number_of_neurons(activations):
        res = {}
        for r in ACTIVATION_RANGES:
            res[r] = count_values_in_range(r, activations)
        return res


    sparsity_per_layer_results, neuron_activation_per_layer = iterate_and_collect_imagenet(
        test_loader, network, result_prefix=result_prefix+'test_')
    sparsity_per_layer_dict = {}
    activations_per_layer_dict = {}
    average_activation_per_neuron_layers = [stat.get_activation_percentage() for stat in neuron_activation_per_layer]
    for sparse_stat in sparsity_per_layer_results:
        sparsity_per_layer_dict[sparse_stat.layer_num] = sparse_stat.average_sparsity

    for idx, average_activation_per_neuron in enumerate(average_activation_per_neuron_layers):
        activations_per_layer_dict[idx] = get_activation_ranges_number_of_neurons(average_activation_per_neuron)
    return sparsity_per_layer_dict, activations_per_layer_dict

# compute sparsity per layer
def iterate_and_collect_imagenet(loader, network, result_prefix=''):
    network.eval()  # put the model in eval mode
    average_sparsity_per_layer = [AverageSparsityStat(i) for i in range(network.depth)]
    neuron_activation_per_layer = [NeuronActivationStat(i) for i in range(network.depth)]
    batch_number = 0
    # collect all activated codes by the data in loader
    with torch.no_grad():
        for data, target in loader:
            output, batch_code_tensor = network.forward_get_code(data)
            batch_number += 1
            pred = [int(out.cpu().detach().numpy())
                    for out in output.data.max(1, keepdim=True)[1]]
            if isinstance(target, torch.Tensor):
                target = target.cpu().detach().numpy()
            if isinstance(target, tuple):
                target = np.asarray(target)
            batch_code_tensor = flatten_list(batch_code_tensor)
            batch_code_numpy = [layer.cpu().detach().numpy()
                                for layer in batch_code_tensor]

            for b in range(batch_code_numpy[0].shape[0]): # One sample at a time
                print(f'Batch number {batch_number}, sample: {b}')
                code_chunks_per_layer = []
                for layer_idx in range(len(batch_code_numpy)):
                    code_chunks_per_layer.append(batch_code_numpy[layer_idx][b].astype(int))
                for l, code_chunk in enumerate(code_chunks_per_layer):
                    average_sparsity_stat = average_sparsity_per_layer[l]
                    neuron_activation_stat = neuron_activation_per_layer[l]
                    neuron_activation_stat.add_sample_code(code_chunk)
                    average_sparsity_stat.add_sample(code_chunk)
    return average_sparsity_per_layer, neuron_activation_per_layer

class NeuronActivationStat():
    def __init__(self, layer_num):
        self.layer_num = layer_num
        self.accumulator = None
        self.count = 0

    def add_sample_code(self, sample):
        self.count += 1
        if self.accumulator is not None:
            self.accumulator = self.accumulator + sample
        else:
            self.accumulator = np.copy(sample)

    def get_activation_percentage(self):
        return self.accumulator / self.count * 100

class AverageSparsityStat():
    def __init__(self, layer_num):
        self.layer_num = layer_num
        self.sample_count = 0
        self.average_sparsity = 0

    def add_sample(self, code_chunk):
        ones_count = np.count_nonzero(code_chunk)
        zeros_count = len(code_chunk) - ones_count
        zeros_percentage = zeros_count / len(code_chunk) * 100
        total_sparsity = self.average_sparsity * self.sample_count + zeros_percentage
        self.sample_count += 1
        self.average_sparsity = total_sparsity / self.sample_count
def count_fraction_code_visited(code_histogram):
    code_example = list(code_histogram.keys())[0]
    # TODO: Fix this so the keys of histograms are consistent (all should be tuples)
    dimension = np.sum([len(layer_code) for layer_code in code_example.split('-')]) if isinstance(code_example, str) else len(code_example)
    total_number_codes = 2 ** dimension
    visited_number_codes = len(list(code_histogram.values()))
    return visited_number_codes / total_number_codes


def combine_all_trials(result_dict):
    combined_results = {}
    first_trial = list(result_dict.keys())[0]
    for metric_key in result_dict[first_trial].keys():
        all_trials = [trial_result[metric_key]
                      for trial_result in result_dict.values()]
        combined_results[metric_key] = all_trials
    return combined_results


def process_results(combined_results):
    processed_result = {}
    for key_metric, trials in combined_results.items():
        if 'histogram' in key_metric:
            all_trials = [count_fraction_code_visited(
                code_histogram) for code_histogram in trials]
            processed_result[key_metric+'_fraction'] = (
                np.mean(all_trials), np.std(all_trials), all_trials)
    return processed_result


def build_latex_table_code_frequency(processed_result):
    def generat_row(end_str, title):
        row = [processed_result['init_train_'+end_str], processed_result['init_test_'+end_str],
               processed_result['post_train_'+end_str], processed_result['post_test_'+end_str]]
        string_row = ['${:2.2f}\pm{:2.2f}$ \%'.format(
            cell[0]*100, cell[1]*100) for cell in row]
        return [title]+string_row
    row_pre = generat_row('code_0_histogram_fraction', 'code prefix')
    row_suf = generat_row('code_1_histogram_fraction', 'code suffix')
    row_code = generat_row('code_histogram_fraction', 'full code')
    rows = [["f", "init", "", "post", ""],
            ['x', 'train', 'test', 'train', 'test'],
            row_pre, row_suf, row_code]

    build_latex_table(rows, "Fraction of visited codes", "table:vis_code")


def compute_suffix_per_prefix(full_code_histogram):
    prefix_key_dict = {}
    layer_size = get_config()['hidden_size']
    for full_code, freq in full_code_histogram.items():
        prefix = full_code[0 : layer_size]
        suffix = full_code[layer_size : len(full_code)]
        if prefix in prefix_key_dict:
            prefix_key_dict[prefix][suffix] = freq
        else:
            prefix_key_dict[prefix] = {suffix: freq}

    return prefix_key_dict


def get_suffix_per_prefix(result_dict):
  # suffix per prefix
    prefix_key_dict = compute_suffix_per_prefix(
        result_dict['post_train_code_histogram'])
    suffix_per_prefix = {}
    for prefix, suffixes in prefix_key_dict.items():
        suffix_per_prefix[prefix] = len(list(suffixes.keys()))
    return suffix_per_prefix

def compute_top_codes(result_dict, NUM_TOP_CODES):
    table_entry_dict = {}
    prefix = 'code_0_histogram'
    suffix = 'code_1_histogram'
    full = 'code_histogram'
    codes_histogram_keys = ['post_train_'+prefix,
                            'post_test_'+prefix,
                            'post_train_'+suffix,
                            'post_test_'+suffix,
                            'post_train_'+full,
                            'post_test_'+full]
    for codes_histogram_key in codes_histogram_keys:
        codes_histogram = result_dict[codes_histogram_key]
        binary_counter = {'0': 0, '1': 0}
        for key, value in codes_histogram.items():
            for c in key:
                c = str(c)
                if c in binary_counter:
                    binary_counter[c] += value

        ratio_1_0 = binary_counter['1']/binary_counter['0']
        frequency = list(codes_histogram.values())
        sorted_index = list(np.argsort(frequency))[-NUM_TOP_CODES:]
        sorted_index.reverse()  # from biggest to smallest
        codes = list(codes_histogram.keys())
        top_codes = [codes[i] for i in sorted_index]
        mass_of_top_codes = [frequency[i] /
                             np.sum(frequency) for i in sorted_index]
        cumulative_mass_of_top_code = list(np.cumsum(mass_of_top_codes))
        table_entry_dict[codes_histogram_key] = {
            'top_codes': top_codes, 'cmass_top_code': cumulative_mass_of_top_code, 'mass_top_code': mass_of_top_codes, 'ratio_1_0': ratio_1_0}
    return table_entry_dict
    
def build_latex_table_top_codes(result_dict, NUM_TOP_CODES=5):
    
    prefix = 'code_0_histogram'
    suffix = 'code_1_histogram'
    full = 'code_histogram'
    codes_histogram_keys = ['post_train_'+prefix,
                            'post_test_'+prefix,
                            'post_train_'+suffix,
                            'post_test_'+suffix,
                            'post_train_'+full,
                            'post_test_'+full]

    table_entry_dict = compute_top_codes(result_dict, NUM_TOP_CODES)
    def create_rows(list_table_entries):
        rows = []
        for i in range(NUM_TOP_CODES):
            row = [str(i)]
            for table_entry in list_table_entries:
                # trick to avoid the int casting that removes the zeros
                code = str(table_entry['top_codes'][i])+':'
                m = '${:2.2f}$ \%'.format(
                    100*table_entry['mass_top_code'][i])
                cm = '${:2.2f}$ \%'.format(
                    100*table_entry['cmass_top_code'][i])
                row = row+[code, m, cm]
            rows.append(row)
        return rows

    title_row = [["x", "train", "", "", "test", "", ""],
                 ['order', 'code', 'mass', 'cumul. mass', 'code', 'mass', 'cumul. mass']]
    full_rows = create_rows(
        [table_entry_dict['post_train_'+full], table_entry_dict['post_test_'+full]])
    build_latex_table(title_row+full_rows, "Top codes of full code",
                      label='fig:top_code')

    prefix_rows = create_rows(
        [table_entry_dict['post_train_'+prefix], table_entry_dict['post_test_'+prefix]])
    build_latex_table(title_row+prefix_rows, "Top codes of prefix",
                      label='fig:prefix_top_code')

    suffix_rows = create_rows(
        [table_entry_dict['post_train_'+suffix], table_entry_dict['post_test_'+suffix]])
    build_latex_table(title_row+suffix_rows, "Top codes of suffix",
                      label='fig:suffix_top_code')


def generate_plots_first_trial(result_dict):
    prefix = 'code_0_histogram'
    suffix = 'code_1_histogram'
    full = 'code_histogram'
    labels = ['init train', 'post train', 'init test', 'post test']

    suffix_per_prefix = get_suffix_per_prefix(result_dict)
    plot_code_histograms([suffix_per_prefix], ['post train'],
                         'suffix_per_prefix', 'suffix_per_prefix_')
    # generate full/prefix/suffix histograms
    plot_code_histograms([result_dict['init_train_'+prefix], result_dict['post_train_'+prefix],
                         result_dict['init_test_'+prefix], result_dict['post_test_'+prefix]], labels, 'prefix', 'prefix_')
    plot_code_histograms([result_dict['init_train_'+suffix], result_dict['post_train_'+suffix],
                          result_dict['init_test_'+suffix], result_dict['post_test_'+suffix]], labels, 'suffix', 'suffix_')
    plot_code_histograms([result_dict['init_train_'+full], result_dict['post_train_'+full],
                         result_dict['init_test_'+full], result_dict['post_test_'+full]], labels, 'full', 'full_')
    plot_code_class_density(
        result_dict['post_train_class_histogram'], 'train', 'train_')
    plot_code_class_density(
        result_dict['post_test_class_histogram'], 'test', 'test_')


def check_results(result_dict):
    
    combined_results = combine_all_trials(result_dict)
    processed_result = process_results(combined_results)

    build_latex_table_code_frequency(processed_result)
    build_latex_table_top_codes(result_dict[0])
    build_latex_table_top_codes(result_dict[0])
    table_entry_dict = compute_top_codes(result_dict[0], NUM_TOP_CODES=5)['post_train_code_histogram']
    codes = table_entry_dict['top_codes']
    for code in codes:
        histo = result_dict[0]['post_train_class_histogram'][code]

        pred = [i[0] for i in histo]
        target = [i[1] for i in histo]
        l_pred = []
        l_target = []
        print(len(histo))
        for c in range(10):
            num_pred = np.where(np.array(pred) == c)[0].shape[0]
            num_target = np.where(np.array(target) == c)[0].shape[0]
            l_pred.append(num_pred)
            l_target.append(num_target)
        acc = np.mean([pred[i] == target[i] for i in range(len(histo))])
        # for c_p in range(10):
        #     for c_t in range(10):

        print(acc)
        print('target',l_target)
        print('pred',l_pred)
    generate_plots_first_trial(result_dict[0])

def check_resnet_results(result_dict, prefix=''):
    num_trials = len(result_dict.keys())
    # combined_results = combine_all_trials(result_dict)
    generate_resnet_sparsity_histo(result_dict[0]['average_sparsity'])
    plot_neuron_activation_percentage_histogram(result_dict[0]['percentage_of_neuron_in_activation_range'])

def generate_resnet_sparsity_histo(result_dict):
    plt.figure(figsize=(15, 10))  # width:20, he
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    ax = plt.figure().gca()
    plt.xlabel('Layer number', fontsize=10)
    plt.ylabel('Percentage of inactive ReLU', fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.bar(range(len(result_dict)), result_dict.values(), align='edge', width=0.5)
    plt.show()

def plot_neuron_activation_percentage_histogram(neurons_in_activation_ranges):
    for layer_idx in range(len(neurons_in_activation_ranges)):
        layer = neurons_in_activation_ranges[layer_idx]
        ranges = []
        frequency = []
        for k, v in layer.items():
            ranges.append(f'({k[0]}% - {k[1]}%)')
            frequency.append(v)
        indices = np.arange(len(ranges))
        plt.bar(indices, frequency, color='r')
        plt.xticks(indices, ranges, rotation='vertical')
        plt.xlabel("Activation frequence range")
        plt.ylabel("Percentage of neurons")
        plt.title(f'Histogram showing percentage of neurons with activation frequency \n falling in denoted range. Layer {layer_idx + 1}')
        plt.tight_layout()
        plt.savefig(f'./figures/neuron_activation_stat_layer{layer_idx}')
        plt.show()
