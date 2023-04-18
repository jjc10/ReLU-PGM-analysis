import math
import networkx
import json
import numpy as np

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
