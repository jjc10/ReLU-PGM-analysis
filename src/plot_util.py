import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score

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


def invert_dict(in_dict):
    out_dict = {}
    for _, value in in_dict.items():
        if value not in out_dict:
            out_dict[value] = 1
        else:
            out_dict[value] += 1
    return out_dict


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
        num_samples = np.sum(values)
        values = [v/num_samples for v in values]
        plt.bar(range(len(values)), values, 1,
                label=labels[i] + ', weighted 1/0 ratio {:2.2f}'.format(binary_counter['1']/binary_counter['0']))
    plt.legend()
    plt.xlabel('codes')
    plt.xlabel('frequency')
    plt.title(prefix_title+' code histograms')
    plt.savefig('figures/'+prefix_file + 'code_histogram.pdf')
    plt.close()
    # generate side plot
    for i, codes_histogram in enumerate(codes_histograms):
        count_histogam = invert_dict(codes_histogram)
        values = list(count_histogam.values())
        values.sort()
        plt.bar(range(len(values)), values, 1, label=labels[i])
    plt.legend()
    plt.xlabel('count')
    plt.xlabel('frequency of codes')
    plt.title(prefix_title+' count histograms')
    plt.savefig('figures/'+prefix_file + 'count_histogram.pdf')
    plt.close()


def get_code_class_hist(result_dict, min_code_occupancy=1, max_code_occupancy=10000, type='true'):
    
        
    key_set = result_dict.keys()
    counts = [0 for i in range(10)]
    accuracies = [[] for i in range(10)]
    for key in key_set:
        if type == 'true':
            tuple_list = [t[0] for t in result_dict[key]]
        elif type == 'predicted':
            tuple_list = [t[1] for t in result_dict[key]]
        
        
        a = accuracy_score([t[0] for t in result_dict[key]], [t[1] for t in result_dict[key]])
        if len(tuple_list) >= min_code_occupancy and len(tuple_list) < max_code_occupancy:
            nunique_codes = len(np.unique(tuple_list))
            counts[nunique_codes-1] += 1
            accuracies[nunique_codes-1].append(a)
    acc = [100*np.mean(a) for a in accuracies]
    return counts, acc


def plot_code_class_density(result_dict, prefix_title, prefix_file):
    def plot(type):
        if type == 'true':
            title = 'true_classes_'
        else:
            title = 'predicted_'
        populations = []
        a = []
        sizes = [1, 2, 5, 10, 50]
        for i in range(1, len(sizes)):
            pop, accs = get_code_class_hist(
                result_dict, min_code_occupancy=sizes[i-1], max_code_occupancy=sizes[i], type=type)
            populations.append(pop)
            a.append(accs)
        pop, accs = get_code_class_hist(
            result_dict, min_code_occupancy=50, type=type)
        a.append(accs)
        print(a)
        populations.append(pop)
        populations = np.array(populations)
        x = np.arange(10) + 1
        plt.bar(x, populations[0])
        plt.bar(x, populations[1], bottom=populations[0])
        plt.bar(x, populations[2], bottom=populations[0]+populations[1])
        plt.bar(x, populations[3], bottom=populations[0] +
                populations[1]+populations[2])
        plt.bar(x, populations[4], bottom=populations[0] +
                populations[1]+populations[2]+populations[3])
        plt.xlabel("Number of unique true classes per code")
        plt.ylabel("Frequency")
        plt.title(prefix_title+'Num. of codes ')
        plt.legend(["10", "11-20", "21-30", "31-40", "50+"],
                   title="Num. of instances in code:")
        plt.xticks(np.arange(0, 11))
        plt.savefig('figures/' + prefix_file + title+"_per_code.pdf")
        plt.close()
    plot(type='true')
    plot(type='predicted')
