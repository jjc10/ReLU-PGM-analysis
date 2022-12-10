from src.table_helper import build_latex_table
import numpy as np
import torch
from src.plot_util import plot_code_histograms, plot_code_class_density


def add_code_histo(histo_dict, key, total, class_per_code_histogram, pred, target):
    if key in histo_dict:
        histo_dict[key] += 1/total
        class_per_code_histogram[key].append((pred, target))
    else:
        histo_dict[key] = 1/total
        class_per_code_histogram[key] = [(pred, target)]


def iterate_and_collect(loader, network, result_prefix=''):
    network.eval()  # put the model in eval mode
    code_histogram = {}
    class_per_code_histogram = {}
    num_datapoints = loader.sampler.num_samples
    code_per_layer_histograms = [{} for _ in range(network.depht)]
    class_per_layer_histograms = [{} for _ in range(network.depht)]
    # collect all activated codes by the data in loader
    with torch.no_grad():
        for data, target in loader:
            output, batch_code_tensor = network.forward_get_code(data)
            pred = [int(out.cpu().detach().numpy())
                    for out in output.data.max(1, keepdim=True)[1]]
            target = target.cpu().detach().numpy()
            batch_code_numpy = [layer.cpu().detach().numpy()
                                for layer in batch_code_tensor]
            for b in range(batch_code_numpy[0].shape[0]):
                code_chunks = [''.join(layer[b, :].astype(int).astype(str))
                               for layer in batch_code_numpy]
                for l, code_chunk in enumerate(code_chunks):
                    layer_code_histogram = code_per_layer_histograms[l]
                    class_per_layer_histogram = class_per_layer_histograms[l]
                    add_code_histo(layer_code_histogram,
                                   code_chunk, num_datapoints, class_per_layer_histogram, pred[b], target[b])
                code = '-'.join(code_chunks)
                add_code_histo(code_histogram,  code,
                               num_datapoints, class_per_code_histogram, pred[b], target[b])

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
    for full_code, freq in full_code_histogram.items():
        code_chunks = full_code.split('-')
        prefix = code_chunks[0]
        suffix = code_chunks[1]
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


def build_latex_table_top_codes(result_dict):
    NUM_TOP_CODES = 5
    prefix = 'code_0_histogram'
    suffix = 'code_1_histogram'
    full = 'code_histogram'
    codes_histogram_keys = ['post_train_'+prefix,
                            'post_test_'+prefix,
                            'post_train_'+suffix,
                            'post_test_'+suffix,
                            'post_train_'+full,
                            'post_test_'+full]

    table_entry_dict = {}
    for codes_histogram_key in codes_histogram_keys:
        codes_histogram = result_dict[codes_histogram_key]
        binary_counter = {'0': 0, '1': 0}
        for key, value in codes_histogram.items():
            for c in key:
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

    def create_rows(list_table_entries):
        rows = []
        for i in range(NUM_TOP_CODES):
            row = [str(i)]
            for table_entry in list_table_entries:
                code = table_entry['top_codes'][i]
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
    plot_code_class_density(result_dict)


def check_results(result_dict, prefix=''):
    num_trials = len(result_dict.keys())
    combined_results = combine_all_trials(result_dict)
    processed_result = process_results(combined_results)

    build_latex_table_code_frequency(processed_result)
    build_latex_table_top_codes(result_dict[0])

    generate_plots_first_trial(result_dict[0])
