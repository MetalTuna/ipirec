# !/bin/bash

# chmod +x ./experiment.sh
# cd /home/taegyu/git_repo/ipitems_analysis/experiments
# pwd
eval "$(conda shell.bash hook)"
# export EXPERIMENT_HOME=/home/tk-hwang/Repositories/ipitems_analysis/experiments
# cd ${EXPERIMENT_HOME}

## [MOVIELENS]
${CONDA_EXE} run -n cfEnv python nmf_ml.py >> nmf_ml_log.txt
${CONDA_EXE} run -n cfEnv python item_cf_ml.py >> item_cf_ml_log.txt

## [COLLEY]
${CONDA_EXE} run -n cfEnv python nmf_colley.py >> nmf_colley_log.txt
${CONDA_EXE} run -n cfEnv python item_cf_colley.py >> item_cf_colley_log.txt

## [IPA]
${CONDA_EXE} run -n cfEnv python ipa_colley.py >> ipa_colley_log.txt
${CONDA_EXE} run -n cfEnv python ipa_ml.py >> ipa_ml_log.txt

#conda activate cfEnv
# python ./item_cf.py >> cmd_log.txt
# conda run -n cfEnv python item_cf.py
# conda deactivate