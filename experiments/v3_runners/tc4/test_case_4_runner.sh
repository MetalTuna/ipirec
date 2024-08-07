# !/bin/bash

for i in {0..4}
do
    ${CONDA_EXE} run -n cfEnv python test_case_4_vlp_v.py $i &
done