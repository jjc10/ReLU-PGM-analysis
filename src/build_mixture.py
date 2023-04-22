from src.config import RESULTS_FOLDER
from src.file_utils import load_most_recent_results, load_most_recent_model
from src.mnist.mnist_analysis import get_mnist_code_frequency
from src.mnist.model_linearization import create_mixture, compute_accuracy_of_mixture, LinearModelMode
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from src.file_utils import get_abs_path_to_results_folder, load_results, load_model

results_folder = get_abs_path_to_results_folder()
# post_train_run_0 = load_most_recent_results(results_folder, "0/post_train.pk")
post_train_run_0 = load_results(f'{results_folder}/mnist_2_6_1682084849', "0/post_train.pk")
# model, config_dict = load_most_recent_model(results_folder)
model, config_dict = load_model(f'{results_folder}/mnist_2_6_1682084849')
# post_train_run_0 = load_most_recent_results(RESULTS_FOLDER, "0/post_train.pk")
result_dict = {"post_train_0": post_train_run_0}
# model, _ = load_most_recent_model(RESULTS_FOLDER)

code_freqs = get_mnist_code_frequency(result_dict)

acc_mix_train = []
acc_mix_test = []
acc_argmax_train = []
acc_argmax_test = []
acc_w_argmax_test = []
acc_w_argmax_train = []

# acc_entropy_train = []
# acc_entropy_test = []


for num_model in tqdm(range(2, 30)):
    mixture_model = create_mixture(model, num_model, code_freqs)
    mix_test_acc, mix_train_acc = compute_accuracy_of_mixture(mixture_model, mode=LinearModelMode.MIXTURE)
    
    acc_mix_test.append(mix_test_acc)
    acc_mix_train.append(mix_train_acc)
    
    argmax_test_acc, argmax_train_acc = compute_accuracy_of_mixture(mixture_model, mode=LinearModelMode.ARGMAX)
    acc_argmax_test.append(argmax_test_acc)
    acc_argmax_train.append(argmax_train_acc)

    w_argmax_test_acc, w_argmax_train_acc = compute_accuracy_of_mixture(mixture_model, mode=LinearModelMode.WEIGHTED_ARGMAX)
    acc_w_argmax_test.append(w_argmax_test_acc)
    acc_w_argmax_train.append(w_argmax_train_acc)
    
    
    # argmax_test_acc, argmax_train_acc = compute_accuracy_of_mixture(mixture_model, mode='Entropy')
    # acc_entropy_test.append(argmax_test_acc)
    # acc_entropy_train.append(argmax_train_acc)
    

# plt.plot(acc_mix_train, label='Mixture train')
plt.figure(figsize=(10, 6), dpi=80)
plt.plot(acc_mix_test, label='Mixture test')
plt.plot(acc_argmax_test, label='Argmax test')
# plt.plot(acc_argmax_train, label='Argmax train')
plt.plot(acc_w_argmax_test, label='Weighted_Argmax test')
# plt.plot(acc_w_argmax_train, label='Weighted_Argmax train')
plt.xticks(np.arange(len(acc_mix_test)), np.arange(2, len(acc_mix_test)+2), rotation='vertical')
# plt.plot(acc_entropy_test, label='Entropy, test')
# plt.plot(acc_entropy_train, label='Entropy, train')
plt.xlabel("Number of linear classifiers used")
plt.ylabel("Percentage accuracy on test set")
plt.legend()
plt.savefig("../figures/mnist_2_6_full_code_perf.pdf")


# plt.show()