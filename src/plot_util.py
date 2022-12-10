import matplotlib.pyplot as plt
import seaborn as sns
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
        values.sort()
        plt.bar(range(len(values)), values, 1,
                label=labels[i] + ', weighted 1/0 ratio {:2.2f}'.format(binary_counter['1']/binary_counter['0']))
    plt.legend()
    plt.xlabel('codes')
    plt.xlabel('frequency')
    plt.title(prefix_title+' code histograms')
    plt.savefig('figures/'+prefix_file + 'code_histogram.pdf')
    plt.close()


def get_code_class_hist(result_dict, min_code_occupancy=1, max_code_occupancy=10000, true_label=0):
    key_set = result_dict['post_train_class_histogram'].keys()
    counts = [0  for i in range(10)]
    for key in key_set:
        tuple_list = []
        for t in result_dict['post_train_class_histogram'][key]:
            tuple_list.append(t[true_label])
        if len(tuple_list) >= min_code_occupancy and len(tuple_list) < max_code_occupancy:
            nunique_codes = len(np.unique(tuple_list))
            counts[nunique_codes-1] += 1
    return counts

def plot_code_class_density(result_dict, sizes = [1, 2, 5, 10, 50], true_label=0):
    populations = []
    for i in range(1, len(sizes)):
        populations.append(get_code_class_hist(result_dict, min_code_occupancy = sizes[i-1], max_code_occupancy = sizes[i]))
    populations.append(get_code_class_hist(result_dict, min_code_occupancy = 50))
    populations = np.array(populations)
    x = np.arange(10) + 1
    plt.bar(x, populations[0])
    plt.bar(x, populations[1], bottom=populations[0])
    plt.bar(x, populations[2], bottom=populations[0]+populations[1])
    plt.bar(x, populations[3], bottom=populations[0]+populations[1]+populations[2])
    plt.bar(x, populations[4], bottom=populations[0]+populations[1]+populations[2]+populations[3])
    plt.xlabel("Number of unique classes per code")
    plt.ylabel("Frequency")
    plt.legend(["1", "2-4", "5-9", "10-49", "50+"], title="Num. of instances in code:")
    plt.xticks(np.arange(0, 11))
    plt.savefig('figures/' + "classes_per_code.pdf")
    plt.close()
