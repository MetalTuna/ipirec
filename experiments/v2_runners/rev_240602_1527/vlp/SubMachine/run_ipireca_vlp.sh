# !/bin/bash
## 태그점수 계산 -> for dtype in DecisionType: [태그점수 보정 -> 개인화] -> 추천

${CONDA_EXE} run -n cfEnv python run_top_n_tags.py &

${CONDA_EXE} run -n cfEnv python run_scores.py &

${CONDA_EXE} run -n cfEnv python run_personalization.py &