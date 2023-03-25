import numpy as np
from collections import OrderedDict
def get_mnist_code_frequency(results):
    post_train = results["post_train_0"]
    total_num_results = len(post_train)
    code_occurrences = count_code_occurrences(post_train)
    code_freqs = count_code_percentage(code_occurrences, total_num_results)
    return code_freqs

# Returns a dictionary of code --> number_of_occurrences
# If a layer is passed, it counts the analysis on that single layer.
def count_code_occurrences(mnist_results_list, layer = None):
    count_dict = {}
    for result in mnist_results_list:
        code = get_full_code(result) if not layer else result.codes[layer] # get full code if no layer is specified, else grab the code for the layer
        if code in count_dict:
            count_dict[code] += 1
        else:
            count_dict[code] = 1
    sorted_descending_dict = OrderedDict(sorted(count_dict.items(), key = lambda kv: kv[1], reverse = True))
    return sorted_descending_dict

def count_code_percentage(code_occurences_dict, total_num_codes):
    frequency_dict = {}
    for k, v in code_occurences_dict.items():
        frequency_dict[k] = v / total_num_codes * 100
    return frequency_dict


def get_full_code(mnist_result):
    full_code = tuple()
    for code in mnist_result.codes:
        full_code += code
    return full_code