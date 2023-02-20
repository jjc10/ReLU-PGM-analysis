from src.config import RESULTS_FOLDER
from file_utils import load_most_recent_results
from src.analysis import check_results, check_resnet_results

result_dict = load_most_recent_results(RESULTS_FOLDER)
try:
    # check_results(result_dict)
    check_resnet_results(result_dict)
except Exception as e:
    print(e)
    print('Could not run, try re-running run_experiment.py')
