
from src.config import RESULTS_FOLDER
from file_utils import load_most_recent_results
from src.analysis import check_results

result_dict = load_most_recent_results(RESULTS_FOLDER)
check_results(result_dict)