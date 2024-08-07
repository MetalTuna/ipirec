# !/bin/bash

${CONDA_EXE} run -n cfEnv python run_top_n_tags.py &

# ${CONDA_EXE} run -n cfEnv python run_scores.py &

${CONDA_EXE} run -n cfEnv python run_personalization.py &