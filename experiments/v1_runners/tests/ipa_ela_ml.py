import os
import sys

WORKSPACE_HOME = os.path.dirname(__file__).replace("/experiments", "")
sys.path.append(WORKSPACE_HOME)
# print(WORKSPACE_HOME)
# exit()

from core import *
from lc_corr import *

if __name__ == "__main__":
    dataset_dir_path = f"{WORKSPACE_HOME}/data/ml"
    testset_file_path = f"{dataset_dir_path}/purchase_list.csv"
    results_summary_path = f"{WORKSPACE_HOME}/results/ipa_ela_ml_topn_ir.csv"

    # 모델의 매개변수들
    dist_n = 1
    learning_rate = 0.01
    generalization = 0.5
    co_occur_items_threshold = 20
    iterations_threshold = 100
    adjust_iterations = 20
    top_n_tags = 10
    top_n_conditions = [n for n in range(3, 21, 2)]

    # 데이터 셋 불러오기
    dataset = MovieLensFilteredDataSet(dataset_dir_path=dataset_dir_path)
    dataset.load_dataset()

    # 모델 구성하기
    model_params = CorrelationModel.create_models_parameters(
        top_n_tags=top_n_tags,
        co_occur_items_threshold=co_occur_items_threshold,
        iterations_threshold=iterations_threshold,
        learning_rate=learning_rate,
    )
    model = CorrelationModel(
        dataset=dataset,
        model_params=model_params,
    )
    model.analysis()

    # 학습하기
    model_params = BiasedEstimator.create_models_parameters(
        learning_rate=learning_rate,
        generalization=generalization,
    )
    estimator = BiasedEstimator(
        model=model,
        model_params=model_params,
    )
    estimator.train(
        DecisionType.E_VIEW,
        n=dist_n,
        emit_iter_condition=adjust_iterations,
    )
    estimator.train(
        DecisionType.E_LIKE,
        n=dist_n,
        emit_iter_condition=adjust_iterations,
    )
    estimator.train(
        DecisionType.E_PURCHASE,
        n=dist_n,
        emit_iter_condition=adjust_iterations,
    )

    # 예측 점수를 기준으로 추천하기
    recommender = ELABasedRecommender(estimator=estimator)
    # recommender = ScoreBasedRecommender(estimator=estimator)
    recommender.prediction()

    # 성능평가하기
    evaluator = IRMetricsEvaluator(
        recommender=recommender,
        file_path=testset_file_path,
    )
    evaluator.top_n_eval(top_n_conditions=top_n_conditions)
    evaluator.evlautions_summary_df().to_csv(path_or_buf=results_summary_path)

# end : main()
