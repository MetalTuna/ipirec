## build in
import os
import sys

__FILE_DIR_PATH = os.path.dirname(__file__)
__dir_name__ = os.path.basename(os.path.dirname(__file__))
WORKSPACE_HOME = __FILE_DIR_PATH.replace(
    f"/experiments/{os.path.basename(__FILE_DIR_PATH)}", ""
)
DATA_SET_HOME = f"{WORKSPACE_HOME}/data"
sys.path.append(WORKSPACE_HOME)

from core import CrossValidationSplitter

if __name__ == "__main__":
    _raw_data_clone_dir_path = f"{WORKSPACE_HOME}/temp/unif_items"
    _kfold_dump_dir_path = f"{DATA_SET_HOME}/colley"
    splitter = CrossValidationSplitter(
        src_dir_path=_raw_data_clone_dir_path,
        dest_dir_path=_kfold_dump_dir_path,
        fold_k=5,
        orded_timestamp=True,
    )
    splitter.split()
# end : main()
