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
    for i, codes_histogram in enumerate(codes_histograms):
        values = list(codes_histogram.values())
        values.sort()
        plt.bar(range(len(values)), values, 1, label=labels[i])
    plt.legend()
    plt.title(prefix_title+' code histograms')
    plt.savefig('figures/'+prefix_file + 'code_histogram.pdf')
    plt.close()