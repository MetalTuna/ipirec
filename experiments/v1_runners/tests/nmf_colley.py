import os
import sys

WORKSPACE_HOME = os.path.dirname(__file__).replace("/experiments", "")
sys.path.append(WORKSPACE_HOME)
# print(WORKSPACE_HOME)
# exit()

from core import *
from lc_corr import *
from decompositions import NMFDecompositionModel, DecompositionsEstimator

if __name__ == "__main__":
    dataset_dir_path = f"{WORKSPACE_HOME}/data/colley"
    testset_file_path = f"{dataset_dir_path}/like_list.csv"
    results_summary_path = f"{WORKSPACE_HOME}/results/nmf_colley_TopN_IRMetrics.csv"

    # 모델의 매개변수들
    factors_dim = 40
    iterations = 10
    dist_n = 1
    learning_rate = 0.01
    generalization = 0.02
    top_n_conditions = [n for n in range(3, 21, 2)]

    # 데이터 셋 불러오기
    dataset = ColleyFilteredDataSet(dataset_dir_path=dataset_dir_path)
    dataset.load_dataset()

    # 모델 구성하기
    model_params = NMFDecompositionModel.create_models_parameters(
        factors_dim=factors_dim,
    )
    model = NMFDecompositionModel(
        dataset=dataset,
        model_params=model_params,
    )
    model.analysis()

    # 학습하기
    model_params = DecompositionsEstimator.create_models_parameters(
        learning_rate=learning_rate,
        generalization=generalization,
    )
    estimator = DecompositionsEstimator(
        model=model,
        model_params=model_params,
    )
    estimator.train(DecisionType.E_VIEW)
    estimator.train(DecisionType.E_LIKE)
    estimator.train(DecisionType.E_PURCHASE)

    # 예측 점수를 기준으로 추천하기
    recommender = ScoreBasedRecommender(estimator=estimator)
    recommender.prediction()

    # 성능평가하기
    evaluator = IRMetricsEvaluator(
        recommender=recommender,
        file_path=testset_file_path,
    )
    evaluator.top_n_eval(top_n_conditions=top_n_conditions)
    evaluator.evlautions_summary_df().to_csv(path_or_buf=results_summary_path)

# end : main()
