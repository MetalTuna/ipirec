# !/bin/bash

for i in {0..4} 
do
    ${CONDA_EXE} run -n cfEnv python prob_estimation_rev.py $i &
done