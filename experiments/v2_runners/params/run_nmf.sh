# !/bin/bash
## [NMF]
# Colley
${CONDA_EXE} run -n cfEnv python nmf_colley.py >> nmf_colley_log.txt &
# MovieLens
${CONDA_EXE} run -n cfEnv python nmf_ml.py >> nmf_ml_log.txt &