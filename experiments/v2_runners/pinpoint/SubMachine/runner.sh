# !/bin/bash

${CONDA_EXE} run -n cfEnv python top_n_tags.py &

${CONDA_EXE} run -n cfEnv python scores.py &

${CONDA_EXE} run -n cfEnv python personalizations_ratio.py &

${CONDA_EXE} run -n cfEnv python personalizations_iter.py &
