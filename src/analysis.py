import math
import numpy as np
import torch
from src.plot_util import plot_code_histograms
import latextable
from texttable import Texttable


def add_code_histo(histo_dict, key, total):
    if key in histo_dict:
        histo_dict[key] += 1/total
    else:
        histo_dict[key] = 1/total


def iterate_and_collect(loader, network):
    network.eval()  # put the model in eval mode
    code_histogram = {}
    num_datapoints = loader.sampler.num_samples
    code_per_layer_histograms = [{} for _ in range(network.depht)]
    # collect all activated codes by the data in loader
    with torch.no_grad():
        for data, target in loader:
            output, batch_code_tensor = network.forward_get_code(data)
            batch_code_numpy = [layer.cpu().detach().numpy()
                                for layer in batch_code_tensor]
            for b in range(batch_code_numpy[0].shape[0]):
                code_chunks = [''.join(layer[b, :].astype(int).astype(str))
                                for layer in batch_code_numpy]
                for l, code_chunk in enumerate(code_chunks):
                    layer_code_histogram = code_per_layer_histograms[l]
                    add_code_histo(layer_code_histogram, code_chunk, num_datapoints)
                code = '-'.join(code_chunks)
                add_code_histo(code_histogram, code, num_datapoints)

    

    results = {'code_histogram': code_histogram,
               'code_per_layer_histograms': code_per_layer_histograms}
    return results


def compile_results(network, test_loader, train_loader):

    # iterate through the test set and check which codes are being activated
    test_results = iterate_and_collect(test_loader, network)
    train_results = iterate_and_collect(train_loader, network)

    test_code_histogram = test_results['code_histogram']
    train_code_histogram = train_results['code_histogram']
    test_code_per_layer_histogram = test_results['code_per_layer_histograms']
    train_code_per_layer_histogram = train_results['code_per_layer_histograms']

    percent_test_unseen_train = 0
    for code, val in test_code_histogram.items():
        if code not in train_code_histogram:
            percent_test_unseen_train += val

    compiled_results = {'test_code_histogram': test_code_histogram,
                        'train_code_histogram': train_code_histogram,
                        'test_code_per_layer_histogram': test_code_per_layer_histogram,
                        'train_code_per_layer_histogram': train_code_per_layer_histogram,
                        'percent_test_unseen_train': percent_test_unseen_train}
    return compiled_results


def count_fraction_code_visited(code_histogram):
    code_example = list(code_histogram.keys())[0]

    dimension = np.sum([len(layer_code)
                        for layer_code in code_example.split('-')])
    total_number_codes = 2**dimension
    visited_number_codes = len(list(code_histogram.values()))
    return total_number_codes, visited_number_codes, visited_number_codes/total_number_codes


def check_results(list_compiled_results, labels, prefix=''):
    table_array = np.zeros((4, 5)).astype(object)
    i = 0
    table_array[0, 0] = '% not in train'
    table_array[1, 0] = 'code prefix'
    table_array[2, 0] = 'code suffix'
    table_array[3, 0] = 'full code'

    def put_entry(row, column, total, visited, fraction):
        table_entry = '(${:.0f}/2^{:.0f}$) {:2.2f} \%'.format(
            visited, math.log2(total), fraction*100)
        table_array[row, column] = table_entry
   
    for compiled_results in list_compiled_results:

        table_array[0, (i*2)+2] = "{:2.2f}\%".format(
            compiled_results['percent_test_unseen_train']*100)

        total, visited, fraction = count_fraction_code_visited(
            compiled_results['train_code_histogram'])
        put_entry(3, (i*2) + 1, total, visited, fraction)

        total, visited, fraction = count_fraction_code_visited(
            compiled_results['test_code_histogram'])
        put_entry(3, (i*2) + 2, total, visited, fraction)

        for d, train_code_layer_histogram in enumerate(compiled_results['train_code_per_layer_histogram']):
            total, visited, fraction = count_fraction_code_visited(
                train_code_layer_histogram)
            put_entry(d+1, (i*2) + 1, total, visited, fraction)

        for d, test_code_layer_histogram in enumerate(compiled_results['test_code_per_layer_histogram']):
            total, visited, fraction = count_fraction_code_visited(
                test_code_layer_histogram)
            put_entry(d+1, (i*2) + 2, total, visited, fraction)
        i += 1
   
    table_1 = Texttable()
    table_1.set_cols_align(["l", "c", "c", "c", "c"])
    rows = [[str(table_entry) for table_entry in table_array[i]] for i in range(4)]
    table_1.add_rows([["f", "pre training", "", "post training", ""],
                     ['x', 'train', 'test', 'train', 'test']]+ rows)
    
    print(table_1.draw())
    print('\nLatextable Output:')
    print(latextable.draw_latex(
        table_1, caption="Fraction of visited codes", label="table:vis_code"))

    train_histograms = [compiled_results['train_code_histogram']
                        for compiled_results in list_compiled_results]
    test_histograms = [compiled_results['test_code_histogram']
                       for compiled_results in list_compiled_results]
    plot_code_histograms(test_histograms, labels,
                         prefix_title=prefix+'test', prefix_file=prefix+'test_')
    plot_code_histograms(train_histograms, labels,
                         prefix_title=prefix+'train', prefix_file=prefix+'training_')
    train_prefix_histograms = [compiled_results['train_code_per_layer_histogram'][0] for compiled_results in list_compiled_results]
    train_suffix_histograms = [compiled_results['train_code_per_layer_histogram'][1] for compiled_results in list_compiled_results]
    test_prefix_histograms = [compiled_results['test_code_per_layer_histogram'][0] for compiled_results in list_compiled_results]
    test_suffix_histograms = [compiled_results['test_code_per_layer_histogram'][1] for compiled_results in list_compiled_results]
    plot_code_histograms(train_prefix_histograms, labels,
                         prefix_title=prefix+'train_prefix', prefix_file=prefix+'train_prefix_')
    plot_code_histograms(train_suffix_histograms, labels,
                         prefix_title=prefix+'train_suffix', prefix_file=prefix+'train_suffix_')
    plot_code_histograms(test_prefix_histograms, labels,
                         prefix_title=prefix+'test_prefix', prefix_file=prefix+'test_prefix_')
    plot_code_histograms(test_suffix_histograms, labels,
                         prefix_title=prefix+'test_suffix', prefix_file=prefix+'test_suffix_')