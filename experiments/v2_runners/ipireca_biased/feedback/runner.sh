# !/bin/bash

${CONDA_EXE} run -n cfEnv python baseline.py &

${CONDA_EXE} run -n cfEnv python approx.py &
