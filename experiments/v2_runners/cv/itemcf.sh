# !/bin/bash
## [MovieLens]
${CONDA_EXE} run -n cfEnv python item_cf_ml.py &
## [Colley]
${CONDA_EXE} run -n cfEnv python item_cf_colley.py &