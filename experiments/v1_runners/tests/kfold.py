import os
import sys

WORKSPACE_HOME = os.path.dirname(__file__).replace("/experiments/tests", "")
sys.path.append(WORKSPACE_HOME)
# print(WORKSPACE_HOME)
# exit()

from core import *
from lc_corr import *

if __name__ == "__main__":
    DATASET_HOME_PATH = dataset_dir_path = f"{WORKSPACE_HOME}/data/"
    k = 5
    dataset_dir_path_dict = dict()

    for dataset_toggle in DataType:
        dataset_dir_path = f"{WORKSPACE_HOME}/data/{DataType.to_str(dataset_toggle)}"
        dataset_dir_path_dict.update({dataset_toggle: dataset_dir_path})
        splitter = CrossValidationSplitter(
            src_dir_path=dataset_dir_path,
            dest_dir_path=dataset_dir_path,
            fold_k=k,
            orded_timestamp=True,
        )
        splitter.split()
    # end : for (DataType = [ Colley, MovieLens ])
# end : main()
