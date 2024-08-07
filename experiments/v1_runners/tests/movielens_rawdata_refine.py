import os
import sys

__dir_path_str__ = os.path.dirname(__file__)
__dir_name__ = os.path.basename(__dir_path_str__)
WORKSPACE_HOME = __dir_path_str__.replace(f"/{__dir_name__}", "")
sys.path.append(WORKSPACE_HOME)

from refine import *

if __name__ == "__main__":
    src_dir_path = f"{WORKSPACE_HOME}/data/ml"
    # dest_dir_path = f"{WORKSPACE_HOME}"
    repo = MovieLensRepository(raw_data_path=src_dir_path)
    repo.convert_decision()
# end : main()
