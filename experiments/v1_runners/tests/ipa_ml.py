import os
import sys

WORKSPACE_HOME = os.path.dirname(__file__).replace("/experiments", "")
sys.path.append(WORKSPACE_HOME)
# print(WORKSPACE_HOME)
# exit()

import numpy as np

from core import *
from lc_corr import *

if __name__ == "__main__":
    dataset_dir_path = f"{WORKSPACE_HOME}/data/ml"
    testset_file_path = f"{dataset_dir_path}/like_list.csv"
    results_summary_path = f"{WORKSPACE_HOME}/results/ipa_ml_TopN_IRMetrics.csv"

    # 모델의 매개변수들
    """
    # test OPT
    self.iterations = 20
    self.dist_num = 2
    self.learning_rate = 0.5
    self.generalization = 0.005
    """

    """
    # init OPT
    dist_n = 1
    learning_rate = 0.01
    generalization = 0.5
    co_occur_items_threshold = 20
    iterations_threshold = 100
    top_n_tags = 10
    top_n_conditions = [n for n in range(3, 21, 2)]
    """

    """
    # heuristic apporach
    dist_n = 1
    learning_rate = 0.0002
    generalization = 0.001
    co_occur_items_threshold = 10
    iterations_threshold = 100
    adjust_iterations = 30
    top_n_tags = 10
    top_n_conditions = [n for n in range(3, 21, 2)]
    """
    """
    dist_n = 2
    learning_rate = 0.0002
    generalization = 0.0001
    co_occur_items_threshold = 1
    iterations_threshold = 30
    adjust_iterations = 10
    top_n_tags = 40
    top_n_conditions = [n for n in range(3, 21, 2)]
    """

    dist_n = 2
    iterations_threshold = 20
    learning_rate = 0.5
    generalization = 0.005
    top_n_tags = 10
    co_occur_items_threshold = 20
    top_n_conditions = [n for n in range(3, 21, 2)]

    # 데이터 셋 불러오기
    dataset = MovieLensFilteredDataSet(dataset_dir_path=dataset_dir_path)
    dataset.load_dataset()

    # 모델 구성하기
    """
    model = CorrelationModel(
        dataset=dataset,
        top_n_tags=top_n_tags,
        co_occur_items_threshold=co_occur_items_threshold,
        iterations_threshold=iterations_threshold,
        learning_rate=learning_rate,
    )
    """
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
        emit_iter_condition=iterations_threshold,
    )
    estimator.train(
        DecisionType.E_LIKE,
        n=dist_n,
        emit_iter_condition=iterations_threshold,
    )
    estimator.train(
        DecisionType.E_PURCHASE,
        n=dist_n,
        emit_iter_condition=iterations_threshold,
    )

    # 예측 점수를 기준으로 추천하기
    recommender = ScoreBasedRecommender(estimator=estimator)
    recommender.prediction()

    # 성능평가하기
    evaluator = IRMetricsEvaluator(
        recommender=recommender,
        file_path=testset_file_path,
    )
    # evaluator.threshold_eval([round(th, 1) for th in np.arange(0.1, 1.1, 0.1)])
    evaluator.top_n_eval(top_n_conditions=top_n_conditions)
    evaluator.evlautions_summary_df().to_csv(path_or_buf=results_summary_path)

# end : main()
