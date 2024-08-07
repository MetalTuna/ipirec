# !/bin/bash
## [ItemCF]
${CONDA_EXE} run -n cfEnv python item_cf_ml_like.py &

${CONDA_EXE} run -n cfEnv python item_cf_ml_purchase.py &
