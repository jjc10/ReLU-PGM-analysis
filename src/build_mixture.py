from src.config import RESULTS_FOLDER
from file_utils import load_most_recent_results, load_most_recent_model
from src.analysis import check_resnet_results
from src.mnist.mnist_analysis import get_mnist_code_frequency
from src.mnist.model_linearization import create_mixture, compute_accuracy_of_mixture


post_train_run_0 = load_most_recent_results(RESULTS_FOLDER, "0/post_train.pk")
result_dict = {"post_train_0": post_train_run_0}
model = load_most_recent_model(RESULTS_FOLDER)

code_freqs = get_mnist_code_frequency(result_dict)
mixture_model = create_mixture(model, 20, code_freqs)
compute_accuracy_of_mixture(mixture_model)