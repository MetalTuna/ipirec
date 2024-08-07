## build in
import os
import sys

# [DEF] Environment variables
__dir_name__ = os.path.basename(os.path.dirname(__file__))
WORKSPACE_HOME = os.path.dirname(__file__).replace(f"/runners/{__dir_name__}", "")
""".../ipirec"""
DATA_SET_HOME = f"{WORKSPACE_HOME}/data"

# [SET] Env.append($WORKSPACE_HOME)
sys.path.append(WORKSPACE_HOME)

from core import *

if __name__ == "__main__":
    _raw_data_dir_path = f"{DATA_SET_HOME}/colley"
    splitter = CrossValidationSplitter(_raw_data_dir_path)
    splitter.split()
# end : main()
