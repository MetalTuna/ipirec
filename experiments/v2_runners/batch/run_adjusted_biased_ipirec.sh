# !/bin/bash
${CONDA_EXE} run -n cfEnv python adjust_biased_ipirec_colley.py >> colley_log.txt &
## pended 
# ${CONDA_EXE} run -n cfEnv python adjust_biased_ipirec_ml.py >> movielens_log.txt &