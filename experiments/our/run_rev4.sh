# !/bin/bash

for i in {0..4}
do
    ${CONDA_EXE} run -n cfEnv python rev4.py $i &
done
