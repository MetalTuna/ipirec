# !/bin/bash

${CONDA_EXE} run -n cfEnv python run_ibcf.py &

${CONDA_EXE} run -n cfEnv python run_nmf.py &

${CONDA_EXE} run -n cfEnv python run_ipirec.py &
