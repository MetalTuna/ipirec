# !/bin/bash

# ${CONDA_EXE} run -n cfEnv python run_top_n_tags.py &

# ${CONDA_EXE} run -n cfEnv python run_scores.py &

# ${CONDA_EXE} run -n cfEnv python run_personalization.py &

## optimal models opt.
# AdjustedBiasedCorrelationEstimator (fit)
# ${CONDA_EXE} run -n cfEnv python run_optima_opt.py &

# AdjustedBiasedCorrelationApproxEstimator
${CONDA_EXE} run -n cfEnv python run_optima_opt_approx.py &