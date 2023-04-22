import math
import networkx
import json
import numpy as np
from functools import reduce
from src.mnist.mnist_experiment import ReluActivationResult

# Numpy arrays are not serializable by default, are you serious?!
# from https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def within_epsilon(a, b, epsilon = 0.0001):
    return math.fabs(a - b) < epsilon

def serialize_networkx_graph(graph, directory, file_name):
    networkx.readwrite.write_gpickle(graph,  f'{directory}/{file_name}.gpickle')

def deserialize_networkx_graph(directory, file_name):
    return networkx.readwrite.read_gpickle(f'{directory}/{file_name}.gpickle')


def convert_boolean_code_to_int(boolean_code):
    binary_string = np.array2string(np.array(boolean_code, dtype=int), separator='')[1:-1] # remove [ and ] at each end.
    return int(binary_string, 2)

def convert_int_to_boolean_code(int, num_bits):
    return f'{int:0{num_bits}b}'

def get_prob_dictionary_from_activation_results(relu_activation_results: list[ReluActivationResult]):
    def reduce_tuple_list(t1, t2):
        return t1 + t2
    count_dict = {}
    for relu_activation_result in relu_activation_results:
        codes = relu_activation_result.codes
        key = reduce(reduce_tuple_list, codes)
        if key in count_dict:
            count_dict[key] += 1
        else:
            count_dict[key] = 1
    total_count = len(relu_activation_results)
    # Normalize
    for k, v in count_dict.items():
        count_dict[k] = v / total_count
    return count_dict