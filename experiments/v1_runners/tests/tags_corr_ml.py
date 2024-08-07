import os
import sys

WORKSPACE_HOME = os.path.dirname(__file__).replace("/experiments", "")
sys.path.append(WORKSPACE_HOME)
# print(WORKSPACE_HOME)
# exit()

import numpy as np

from core import *
from lc_corr import *
from rec_tags_freq import *

if __name__ == "__main__":
    dataset_dir_path = f"{WORKSPACE_HOME}/data/ml"
    # testset_file_path = f"{dataset_dir_path}/like_list.csv"
    testset_file_path = f"{dataset_dir_path}/purchase_list.csv"
    results_summary_path = f"{WORKSPACE_HOME}/results/ipa_ml_TopN_IRMetrics.csv"
    raw_corr_figure_file_path = f"{WORKSPACE_HOME}/results/ipa_ml_raw_tags_corr.svg"
    biased_corr_figure_file_path = (
        f"{WORKSPACE_HOME}/results/ipa_ml_biased_tags_corr.svg"
    )
    corr_weight_figure_file_path = (
        f"{WORKSPACE_HOME}/results/ipa_ml_tags_corr_wegiht.svg"
    )
    adjusted_corr_figure_file_path = (
        f"{WORKSPACE_HOME}/results/ipa_ml_adjuted_tags_corr.svg"
    )

    # 모델의 매개변수들
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

    # [DRAW] heatmap plot
    HeatmapFigure.draw_heatmap(
        tag_idx_to_name_dict=model.tag_idx_to_name,
        tags_corr=model.arr_tags_score,
        fig_title="raw_tags_score",
        file_path=raw_corr_figure_file_path,
    )

    # 학습하기
    model_params = BiasedEstimator.create_models_parameters(
        learning_rate=learning_rate,
        generalization=generalization,
    )
    estimator = BiasedEstimator(
        model=model,
        model_params=model_params,
    )

    # [DRAW] heatmap plot
    HeatmapFigure.draw_heatmap(
        tag_idx_to_name_dict=model.tag_idx_to_name,
        tags_corr=estimator.model.arr_tags_score,
        fig_title="biased_tags_score",
        file_path=biased_corr_figure_file_path,
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
    # [DRAW] heatmap plot
    tags_weight: np.ndarray = estimator.arr_user_idx_to_weights.mean(axis=0)
    HeatmapFigure.draw_heatmap(
        tag_idx_to_name_dict=model.tag_idx_to_name,
        tags_corr=tags_weight,
        fig_title="tags_score_weight",
        file_path=corr_weight_figure_file_path,
        diag_score=1.0,
    )
    weighted_tags_corr = np.multiply(
        estimator.arr_tags_score, tags_weight
    )  # estimator.arr_tags_score * tags_weight
    HeatmapFigure.draw_heatmap(
        tag_idx_to_name_dict=model.tag_idx_to_name,
        tags_corr=weighted_tags_corr,
        fig_title="adjusted_tags_score",
        file_path=adjusted_corr_figure_file_path,
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

    tags_freq_dist = CosineItemsTagsFreq()
    distance: float = tags_freq_dist.tags_freq_distance(
        test_set=evaluator.TEST_SET_LIST,
        recommender=recommender,
    )

    print(f"tags freq. distance (cosine): {distance}")
# end : main()
