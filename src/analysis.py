import numpy as np
import torch
from src.plot_util import plot_code_histograms


def add_code_histo(histo_dict, key, total):
    if key in histo_dict:
        histo_dict[key] += 1/total
    else:
        histo_dict[key] = 1/total


def iterate_and_collect(loader, network):
    network.eval()  # put the model in eval mode
    code_histogram = {}
    num_datapoints = loader.sampler.num_samples

    # collect all activated codes by the data in loader
    with torch.no_grad():
        for data, target in loader:
            output, batch_code_tensor = network.forward_get_code(data)
            batch_code_numpy = [layer.cpu().detach().numpy()
                                for layer in batch_code_tensor]
            for b in range(batch_code_numpy[0].shape[0]):
                code = '-'.join([''.join(layer[b, :].astype(int).astype(str))
                                for layer in batch_code_numpy])
                add_code_histo(code_histogram, code, num_datapoints)

    code_per_layer_histograms = [{} for _ in range(network.depht)]
    # each layer's code is separated by -
    # look at each layer's code separately
    for key, value in code_histogram.items():
        codes_per_layer = key.split('-')
        for layer_index, layer_code in enumerate(codes_per_layer):
            layer_code_histogram = code_per_layer_histograms[layer_index]
            add_code_histo(layer_code_histogram, key=layer_code, total=1/value)

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


def check_results(compiled_results, prefix):
    print("Percent of test code that are not in train codes {:2.2f}%".format(
        compiled_results['percent_test_unseen_train']*100))
    plot_code_histograms(
        compiled_results['test_code_histogram'], prefix_title=prefix+'test', prefix_file=prefix+'test_')
    plot_code_histograms(
        compiled_results['train_code_histogram'], prefix_title=prefix+'train', prefix_file=prefix+'training_')

    total, visited, fraction = count_fraction_code_visited(
        compiled_results['test_code_histogram'])
    print('Code visited of test samples', visited,
          'over', total, ' ; {:2.2f} %'.format(fraction*100))

    total, visited, fraction = count_fraction_code_visited(
        compiled_results['train_code_histogram'])
    print('Code visited of train samples', visited,
          'over', total, ' ; {:2.2f} %'.format(fraction*100))

    for d, train_code_layer_histogram in enumerate(compiled_results['train_code_per_layer_histogram']):

        plot_code_histograms(train_code_layer_histogram, prefix_title=prefix+'train layer '+str(d),
                             prefix_file=prefix+'training_layer_'+str(d)+'_')
        total, visited, fraction = count_fraction_code_visited(
            train_code_layer_histogram)
        print('Code visited of train samples layer', d, visited,
              'over', total, ' ; {:2.2f} %'.format(fraction*100))

    for d, test_code_layer_histogram in enumerate(compiled_results['test_code_per_layer_histogram']):

        plot_code_histograms(test_code_layer_histogram, prefix_title=prefix+'test layer '+str(d),
                             prefix_file=prefix+'testing_layer_'+str(d)+'_')
        total, visited, fraction = count_fraction_code_visited(
            test_code_layer_histogram)
        print('Code visited of test samples layer', d, visited,
              'over', total, ' ; {:2.2f} %'.format(fraction*100))
