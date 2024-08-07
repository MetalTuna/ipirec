# Build-in
import os
import sys

__FILE_DIR_PATH = os.path.dirname(__file__)
__dir_name__ = os.path.basename(os.path.dirname(__file__))
WORKSPACE_HOME = __FILE_DIR_PATH.replace(
    f"/experiments/{os.path.basename(__FILE_DIR_PATH)}", ""
)
DATA_SET_HOME = f"{WORKSPACE_HOME}/data"
sys.path.append(WORKSPACE_HOME)

# Custom LIB.
from core import Machine
from refine import ColleyDatesItemsReductionRepository

if __name__ == "__main__":
    _raw_data_clone_dir_path = f"{WORKSPACE_HOME}/temp/unif_items"
    repo = ColleyDatesItemsReductionRepository(
        _raw_data_clone_dir_path,
        db_src=Machine.E_MAC,
        begin_date_str="2023-11-01",
        emit_date_str="2023-12-31",
    )
    # repo.items_reduction(positive_threshold=65)
    # repo.items_reduction(positive_threshold=75)
    repo.items_reduction(positive_threshold=40)
# end : main()
