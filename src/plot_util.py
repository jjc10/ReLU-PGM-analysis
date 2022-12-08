import matplotlib.pyplot as plt
import numpy as np


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


def plot_code_histograms(codes_histograms, labels, prefix_title, prefix_file):
    binary_counter = {'0': 0, '1': 0}
    for i, codes_histogram in enumerate(codes_histograms):
        # count the fraction of 1 and 0
        for key, value in codes_histogram.items():
            for c in key:
                if c in binary_counter:
                    binary_counter[c] += value
        values = list(codes_histogram.values())
        sorted_index = list(np.argsort(values))[-2:]
        codes = list(codes_histogram.keys())
        top_code = [codes[i] for i in sorted_index]
        values.sort()
        plt.bar(range(len(values)), values, 1,
                label=labels[i] + ', weighted 1/0 ratio {:2.2f}'.format(binary_counter['1']/binary_counter['0']))
    plt.legend()
    plt.xlabel('codes')
    plt.xlabel('frequency')
    plt.title(prefix_title+' code histograms')
    plt.savefig('figures/'+prefix_file + 'code_histogram.pdf')
    plt.close()
