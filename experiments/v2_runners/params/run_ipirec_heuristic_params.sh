# !/bin/bash
## [IPIRec]
# Colley
${CONDA_EXE} run -n cfEnv python ipirec_params_heuristics_colley.py &
# MovieLens
${CONDA_EXE} run -n cfEnv python ipirec_params_heuristics_ml.py &