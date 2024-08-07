###
# 24.05.23 12:04 Observ. Corr. Res.
###


### buildt-in
import os
import sys

### 3rd Pty.
import numpy as np

### Custom LIB
__CURRRENT_DIR_PATH = os.path.dirname(__file__)
WORKSPACE_HOME = __CURRRENT_DIR_PATH.replace(
    f"/experiments/{os.path.basename(__CURRRENT_DIR_PATH)}", ""
)
sys.path.append(WORKSPACE_HOME)
# print(WORKSPACE_HOME)
# exit()

from core import *
from colley import *
from ipirec import *
from movielens import *
from rec_tags_freq import *

if __name__ == "__main__":
    selected_data: DataType = None
    selected_decision: DecisionType = None
    dataset: BaseDataSet = None
    model: BaseModel = None
    estimator: BaseEstimator = None
    recommender: BaseRecommender = None
    evaluator: BaseEvaluator = None

    model_str = "BiasedIPIRecA"
    selected_data = DataType.E_COLLEY
    # selected_decision = DecisionType.E_LIKE
    selected_decision = DecisionType.E_PURCHASE

    KFold = 5
    top_n_tags = 10
    co_occur_items_threshold = 4
    score_iter = 3
    score_learning_rate = 0.1
    score_generalization = 1.0
    weight_iter = 5
    weight_learning_rate = 0.01
    weight_generalization = 0.1
    frob_norm = 1
    default_voting = 0.0

    top_n_conditions = [n for n in range(3, 37, 2)]

    dataset_str = DataType.to_str(selected_data)
    decision_str = DecisionType.to_str(selected_decision)
    dataset_dir_path = f"{WORKSPACE_HOME}/data/{dataset_str}"
    # testset_file_path = f"{dataset_dir_path}/{decision_str}_list.csv"
    results_dir_path = f"{WORKSPACE_HOME}/results/tags_dist"

    opt_file_name_str = f"{model_str}_{dataset_str}"
    results_summary_path = f"{results_dir_path}/{opt_file_name_str}_TopN_IRMetrics.csv"

    __dir_path = os.path.dirname(results_summary_path)
    if not DirectoryPathValidator.exist_dir(__dir_path):
        DirectoryPathValidator.mkdir(__dir_path)

    tags_freq_dist_dict = dict()
    """
    Key: set_id (int)
    Value: distance (float)
    """

    ### figures_file_name_sty: ${RESULTS_HOME_PATH}/${MODEL_OPT}_${ELEMENTS}_${SET_NO}_${DECISION} ###
    """
    raw_corr_figure_file_path = (
        f"{results_dir_path}/{opt_file_name_str}_raw_tags_corr_{k}_{decision_str}.svg"
    )
    biased_corr_figure_file_path = f"{results_dir_path}/{opt_file_name_str}_biased_tags_corr_{k}_{decision_str}.svg"
    corr_weight_figure_file_path = f"{results_dir_path}/{opt_file_name_str}_tags_corr_wegiht_{k}_{decision_str}.svg"
    adjusted_corr_figure_file_path = f"{results_dir_path}/{opt_file_name_str}_adjuted_tags_corr_{k}_{decision_str}.svg"
    """

    for k in range(KFold):
        raw_corr_figure_file_path = f"{results_dir_path}/{opt_file_name_str}_raw_tags_corr_{k}_{decision_str}.svg"
        biased_corr_figure_file_path = f"{results_dir_path}/{opt_file_name_str}_biased_tags_corr_{k}_{decision_str}.svg"
        corr_weight_figure_file_path = f"{results_dir_path}/{opt_file_name_str}_tags_corr_weight_{k}_{decision_str}.svg"
        adjusted_corr_figure_file_path = f"{results_dir_path}/{opt_file_name_str}_adjuted_tags_corr_{k}_{decision_str}.svg"

        match (selected_data):
            case DataType.E_COLLEY:
                dataset = ColleyFilteredDataSet(dataset_dir_path)
            case DataType.E_MOVIELENS:
                dataset = MovieLensFilteredDataSet(dataset_dir_path)
            case _:
                raise NotImplementedError()
        # end : match-case

        dataset._load_metadata_()
        for decision_type in DecisionType:
            file_path = str.format(
                "{0}/train_{1}_{2}_list.csv",
                dataset_dir_path,
                k,
                DecisionType.to_str(decision_type),
            )
            dataset.append_decisions(
                file_path=file_path,
                decision_type=decision_type,
            )
        # end : for (DecisionTypes)
        dataset.__id_index_mapping__()

        model_params = CorrelationModel.create_models_parameters(
            top_n_tags=top_n_tags,
            co_occur_items_threshold=co_occur_items_threshold,
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
        model_params = AdjustedBiasedCorrelationEstimator.create_models_parameters(
            score_iterations=score_iter,
            score_learning_rate=score_learning_rate,
            score_generalization=score_generalization,
            weight_iterations=weight_iter,
            weight_learning_rate=weight_learning_rate,
            weight_generalization=weight_generalization,
            frob_norm=frob_norm,
            default_voting=default_voting,
        )
        estimator = AdjustedBiasedCorrelationEstimator(
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

        ## Training -- Personalized
        # VIEW
        target_dtype = DecisionType.E_VIEW
        kwd = DecisionType.to_str(target_dtype)
        estimator.train(
            target_decision=target_dtype,
            n=frob_norm,
            emit_iter_condition=weight_iter,
        )
        HeatmapFigure.draw_heatmap(
            tag_idx_to_name_dict=model.tag_idx_to_name,
            tags_corr=estimator.arr_tags_score,
            fig_title=f"trained_{kwd}",
            file_path=f"{results_dir_path}/{opt_file_name_str}_trained_{kwd}_{k}_{decision_str}.svg",
            diag_score=0.0,
        )

        # LIKE
        target_dtype = DecisionType.E_LIKE
        kwd = DecisionType.to_str(target_dtype)
        estimator.train(
            target_decision=target_dtype,
            n=frob_norm,
            emit_iter_condition=weight_iter,
        )
        HeatmapFigure.draw_heatmap(
            tag_idx_to_name_dict=model.tag_idx_to_name,
            tags_corr=estimator.arr_tags_score,
            fig_title=f"trained_{kwd}",
            file_path=f"{results_dir_path}/{opt_file_name_str}_trained_{kwd}_{k}_{decision_str}.svg",
            diag_score=0.0,
        )

        # PURCHASE
        target_dtype = DecisionType.E_PURCHASE
        kwd = DecisionType.to_str(target_dtype)
        estimator.train(
            target_decision=target_dtype,
            n=frob_norm,
            emit_iter_condition=weight_iter,
        )
        HeatmapFigure.draw_heatmap(
            tag_idx_to_name_dict=model.tag_idx_to_name,
            tags_corr=estimator.arr_tags_score,
            fig_title=f"trained_{kwd}",
            file_path=f"{results_dir_path}/{opt_file_name_str}_trained_{kwd}_{k}_{decision_str}.svg",
            diag_score=0.0,
        )

        tags_weight: np.ndarray = estimator.arr_user_idx_to_weights.mean(axis=0)
        weighted_tags_corr = np.multiply(
            estimator.arr_tags_score, tags_weight
        )  # estimator.arr_tags_score * tags_weight
        HeatmapFigure.draw_heatmap(
            tag_idx_to_name_dict=model.tag_idx_to_name,
            tags_corr=model.arr_tags_score,
            fig_title="trained_tags_score",
            file_path=adjusted_corr_figure_file_path,
        )

        # 예측 점수를 기준으로 추천하기
        recommender = ScoreBasedRecommender(estimator=estimator)
        recommender.prediction()

        # 성능평가하기
        file_path = str.format(
            "{0}/test_{1}_{2}_list.csv",
            dataset_dir_path,
            k,
            decision_str,
        )
        evaluator = IRMetricsEvaluator(
            recommender=recommender,
            file_path=file_path,
        )
        evaluator.top_n_eval(top_n_conditions=top_n_conditions)
        evaluator.evlautions_summary_df().to_csv(path_or_buf=results_summary_path)

        # tags_freq_dist = CosineItemsTagsFreq()
        tags_freq_dist = CosineItemsTagsFreqAddPenalty()
        distance: float = tags_freq_dist.tags_freq_distance(
            test_set=evaluator.TEST_SET_LIST,
            recommender=recommender,
        )
        tags_freq_dist_dict.update({k: distance})
        print(f"tags freq. distance (cosine) [{k}/{KFold}]: {distance}")

    # end : for (KFold)

    numer = 0.0
    for k, distance in tags_freq_dist_dict.items():
        numer += distance
        print(f"Set {k}: {distance}\n")
    # end : for (KFold)
    numer = numer / len(tags_freq_dist_dict)
    print(f"Mean: {numer}")
# end : main()
