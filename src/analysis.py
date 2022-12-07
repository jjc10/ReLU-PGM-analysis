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


def iterate_and_collect(loader, network, result_prefix=''):
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
                    add_code_histo(layer_code_histogram,
                                   code_chunk, num_datapoints)
                code = '-'.join(code_chunks)
                add_code_histo(code_histogram, code, num_datapoints)

    results = {result_prefix+'code_histogram': code_histogram}
    for l, code_per_layer_histogram in enumerate(code_per_layer_histograms):
        results[result_prefix+'code_' +
                str(l)+'_histogram'] = code_per_layer_histogram
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


def count_fraction_code_visited(code_histogram):
    code_example = list(code_histogram.keys())[0]

    dimension = np.sum([len(layer_code)
                        for layer_code in code_example.split('-')])
    total_number_codes = 2**dimension
    visited_number_codes = len(list(code_histogram.values()))
    return visited_number_codes/total_number_codes


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


def build_latex_table(processed_result):
    table_array = np.zeros((4, 5)).astype(object)

    def generat_row(end_str, title):
        row = [processed_result['init_train_'+end_str], processed_result['init_test_'+end_str],
               processed_result['post_train_'+end_str], processed_result['post_test_'+end_str]]
        string_row = ['{:2.2f} $\pm$ {:2.2f} \%'.format(
            cell[0]*100, cell[1]*100) for cell in row]
        return [title]+string_row
    row_pre = generat_row('code_0_histogram_fraction', 'code prefix')
    row_suf = generat_row('code_1_histogram_fraction', 'code suffix')
    row_code = generat_row('code_histogram_fraction', 'full code')
    table_1 = Texttable()
    table_1.set_cols_align(["l", "c", "c", "c", "c"])

    table_1.add_rows([["f", "init", "", "post", ""],
                     ['x', 'train', 'test', 'train', 'test'],
                     row_pre, row_suf, row_code])

    print(table_1.draw())
    print('\nLatextable Output:')
    print(latextable.draw_latex(
        table_1, caption="Fraction of visited codes", label="table:vis_code"))


def generate_plots_first_trial(result_dict):
    pass

def check_results(result_dict, prefix=''):
    num_trials = len(result_dict.keys())
    combined_results = combine_all_trials(result_dict)
    processed_result = process_results(combined_results)

    build_latex_table(processed_result)
    generate_plots_first_trial(result_dict[0])
