# !/bin/bash

# for ((i = 0; i <= 4; i++))
for i in {0..4}
do
    ${CONDA_EXE} run -n cfEnv python vlp_v.py $i
done

for i in {0..4}
do
    ${CONDA_EXE} run -n cfEnv python vlp_l.py $i
done

for i in {0..4}
do
    ${CONDA_EXE} run -n cfEnv python vlp_p.py $i
done