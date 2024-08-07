# Build-in
import os
import sys

UNDEF_IDS = -1
__FILE_DIR_PATH = os.path.dirname(__file__)
WORKSPACE_HOME = __FILE_DIR_PATH.replace(
    f"/runners/dataset_builders/{os.path.basename(__FILE_DIR_PATH)}", ""
)
# sys.path.append(WORKSPACE_HOME)
DATASET_DIR_HOME = f"{WORKSPACE_HOME}/data/colley"
print(WORKSPACE_HOME)
sys.path.append(WORKSPACE_HOME)

# Custom LIB.
from core import Machine
from refine import *

if __name__ == "__main__":
    _debug_dir_path = f"{WORKSPACE_HOME}/temp/unif_items"
    repo = ColleyDatesRepository(
        _debug_dir_path,
        db_src=Machine.E_MAC,
        begin_date_str="2023-12-21",
        emit_date_str="2023-12-31",
    )
    repo.load_data()
    repo.convert_decision()
    repo.dump_data()
# end : main()
