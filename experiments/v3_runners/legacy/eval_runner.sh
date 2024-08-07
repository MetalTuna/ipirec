# !/bin/bash

# ${CONDA_EXE} run -n cfEnv python results_summary.py begin &

# ${CONDA_EXE} run -n cfEnv python results_summary.py gen_v &

# ${CONDA_EXE} run -n cfEnv python results_summary.py gen_p &

${CONDA_EXE} run -n cfEnv python results_summary.py gen_l &

# ${CONDA_EXE} run -n cfEnv python results_summary.py notPostProcess &
