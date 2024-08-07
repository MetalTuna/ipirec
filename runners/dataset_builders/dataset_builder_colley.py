# Build-in
import os
import sys

UNDEF_IDS = -1
__FILE_DIR_PATH = os.path.dirname(__file__)
WORKSPACE_HOME = __FILE_DIR_PATH.replace(
    f"/runners/{os.path.basename(__FILE_DIR_PATH)}", ""
)
# sys.path.append(WORKSPACE_HOME)
DATASET_DIR_HOME = f"{WORKSPACE_HOME}/data/colley"
print(WORKSPACE_HOME)
sys.path.append(WORKSPACE_HOME)

# Custom LIB.
from core import Machine
from refine import *

if __name__ == "__main__":
    """
    _debug_dir_path = f"{WORKSPACE_HOME}/temp/unif_items"
    repo = ColleyDatesItemsReductionRepository(
        _debug_dir_path,
        db_src=Machine.E_MAC,
        begin_date_str="2023-07-01",
        emit_date_str="2023-12-31",
    )
    repo.items_reduction(positive_threshold=65)
    """

    _debug_dir_path = f"{WORKSPACE_HOME}/temp/unif_items"
    repo = ColleyDatesItemsReductionRepository(
        _debug_dir_path,
        db_src=Machine.E_MAC,
        begin_date_str="2023-11-01",
        emit_date_str="2023-12-31",
    )
    repo.items_reduction(positive_threshold=40)
# end : main()
