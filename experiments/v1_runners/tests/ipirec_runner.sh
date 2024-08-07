# !/bin/bash
# eval "$(conda shell.bash hook)"
## [IPIRec] colley
${CONDA_EXE} run -n cfEnv python ipirec_colley_like_params.py >> IPIRec_Colley_like_prompt.txt &

${CONDA_EXE} run -n cfEnv python ipirec_colley_purchase_params.py >> IPIRec_Colley_purchase_prompt.txt &
