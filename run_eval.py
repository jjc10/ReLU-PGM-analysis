from src.config import RESULTS_FOLDER
from file_utils import load_most_recent_results
from src.analysis import check_resnet_results
from src.mnist.mnist_analysis import perform_mnist_analysis



post_train_run_0 = load_most_recent_results(RESULTS_FOLDER, "0/post_train.pk")
pre_train_run_0 = load_most_recent_results(RESULTS_FOLDER, "0/init_train.pk")
result_dict = {"post_train_0": post_train_run_0, "pre_train_run_0": pre_train_run_0}


# check_resnet_results(result_dict)
perform_mnist_analysis(result_dict)