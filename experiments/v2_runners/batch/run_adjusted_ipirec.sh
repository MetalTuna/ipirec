# !/bin/bash
${CONDA_EXE} run -n cfEnv python adjust_ipirec_colley.py &

${CONDA_EXE} run -n cfEnv python adjust_ipirec_ml.py &