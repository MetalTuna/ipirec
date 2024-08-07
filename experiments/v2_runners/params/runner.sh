# !/bin/bash
## [ItemCF]
${CONDA_EXE} run -n cfEnv python itemcf_eval.py >> itemcf_log.txt &
## [IPIRec]
${CONDA_EXE} run -n cfEnv python ipirec_params.py >> ipirec_log.txt &