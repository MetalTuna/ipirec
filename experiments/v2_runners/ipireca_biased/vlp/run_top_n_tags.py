"""
[작성일] 24.05.21 15:58
IPIRec - Adjusted module test

[수정일] 
- 24.05.27 12:00. 모델변수 교차고정하며 관측목적 -- IPIRecA (Biased)
- 24.25.24 16:58. 결과관측을 위한 모델변수 추가(SubMachine), 결과저장 경로변경($RESULTS/observ)
- 24.05.22 19:30. 결과관측을 위한 모델변수 추가(3090)
"""

## build-in
import os
import sys

# [DEF] Environment variables
__dir_name__ = os.path.basename(os.path.dirname(__file__))
WORKSPACE_HOME = os.path.dirname(__file__).replace(
    f"/experiments/ipireca_biased/{__dir_name__}", ""
)
""".../ipirec"""
DATA_SET_HOME = f"{WORKSPACE_HOME}/data"
""">>> `${WORKSPACE_HOME}`/data"""
MODEL_NAME = "IPIRecAB_TopNTags"
RESULTS_SUMMARY_HOME = f"{WORKSPACE_HOME}/results/params/vlp"
""">>> `${WORKSPACE_HOME}`/results/params/vlp"""

# [SET] Env.append($WORKSPACE_HOME)
sys.path.append(WORKSPACE_HOME)

## custom modules
from core import *
from colley import *
from ipirec import *
from movielens import *


def build_models(
    selected_data: DataType,
    default_voting_score: float,
    top_n_tags: int,
    frob_norm_n: int,
    co_occur_items: int,
    score_iterations: int,
    score_learning_rate: float,
    score_generalization: float,
    weight_iterations: int,
    weight_learning_rate: float,
    weight_generalization: float,
):
    data_name = DataType.to_str(selected_data)
    KFold = 5
    top_n_conditions_list = [n for n in range(3, 37, 2)]
    dataset_dir_path = f"{DATA_SET_HOME}/{data_name}"

    for k in range(KFold):
        ## [DEF] models
        model_info_str = ""
        dataset: BaseDataSet = None
        model: BaseModel = None
        estimator: BaseEstimator = None
        recommender: BaseRecommender = None
        evaluator: BaseEvaluator = None

        # [IO] Read dataset
        match (selected_data):
            case DataType.E_MOVIELENS:
                dataset = MovieLensFilteredDataSet(
                    dataset_dir_path=dataset_dir_path,
                )
            case DataType.E_COLLEY:
                dataset = ColleyFilteredDataSet(
                    dataset_dir_path=dataset_dir_path,
                )
        # end : match-case
        dataset._load_metadata_()
        for kwd in DecisionType:
            file_path = file_path = (
                f"{dataset_dir_path}/train_{k}_{DecisionType.to_str(kwd)}_list.csv"
            )
            dataset.append_decisions(
                file_path=file_path,
                decision_type=kwd,
            )
        dataset.__id_index_mapping__()

        model_params = CorrelationModel.create_models_parameters(
            top_n_tags=top_n_tags,
            co_occur_items_threshold=co_occur_items,
        )
        model = CorrelationModel(
            dataset=dataset,
            model_params=model_params,
        )
        model.analysis()
        estimator_params = AdjustedBiasedCorrelationEstimator.create_models_parameters(
            score_iterations=score_iterations,
            score_learning_rate=score_learning_rate,
            score_generalization=score_generalization,
            weight_iterations=weight_iterations,
            weight_learning_rate=weight_learning_rate,
            weight_generalization=weight_generalization,
            frob_norm=frob_norm_n,
            default_voting=default_voting_score,
        )

        estimator = AdjustedBiasedCorrelationEstimator(
            model=model,
            model_params=estimator_params,
        )

        for decision_type in [
            DecisionType.E_VIEW,
            DecisionType.E_LIKE,
            DecisionType.E_PURCHASE,
        ]:
            estimator.train(
                target_decision=decision_type,
                n=frob_norm_n,
                emit_iter_condition=weight_iterations,
            )
        # end : for (decision_types)

        recommender = ScoreBasedRecommender(
            estimator=estimator,
        )
        recommender.prediction()

        for target_decision in [
            DecisionType.E_VIEW,
            DecisionType.E_LIKE,
            DecisionType.E_PURCHASE,
        ]:
            decision_kwd = DecisionType.to_str(target_decision)
            results_summary_path = str.format(
                "{0}/{1}/{2}_TopN_{3}_{4}_IRMetrics_{5}.csv",
                RESULTS_SUMMARY_HOME,
                MODEL_NAME,
                data_name,
                decision_kwd,
                (k + 1),
                DirectoryPathValidator.current_datetime_str(),
            )
            __dir_path_str = os.path.dirname(results_summary_path)
            if not DirectoryPathValidator.exist_dir(__dir_path_str):
                DirectoryPathValidator.mkdir(__dir_path_str)

            model_info_str = ""
            model_info_str += "[DataSet]\n"
            model_info_str += f"Target,{data_name},{decision_kwd}\n"
            model_info_str += f"No.,{k + 1},{KFold}\n"
            model_info_str += "[Model]\n"
            model_info_str += f"top_n_tag: {top_n_tags}\n"
            model_info_str += f"co_occur_items: {co_occur_items}\n"
            model_info_str += f"score_iterations: {score_iterations}\n"
            # model_info_str += f"score_generanlization: {score_generalization}\n"  appended
            model_info_str += f"score_learing_rate: {score_learning_rate}\n"
            model_info_str += "[Estimator]\n"
            model_info_str += f"weight_learning_rate: {weight_learning_rate}\n"
            model_info_str += f"weight_generalization: {weight_generalization}\n"
            model_info_str += f"weight_iteration: {weight_iterations}\n"
            model_info_str += f"Frob_norm: {frob_norm_n}\n"
            model_info_str += f"default_voting: {default_voting_score}\n"
            model_info_str += f"score_generanlization: {score_generalization}\n"

            with open(
                file=results_summary_path,
                mode="at",
                encoding="utf-8",
            ) as fout:
                fout.write(model_info_str)
                fout.close()
            test_file_path = f"{dataset_dir_path}/test_{k}_{decision_kwd}_list.csv"
            evaluator = IRMetricsEvaluator(
                recommender=recommender,
                file_path=test_file_path,
            )
            evaluator.top_n_eval(
                top_n_conditions=top_n_conditions_list,
            )
            evaluator.save_evaluations_summary(
                file_path=results_summary_path,
                mode="at",
            )
    # end : KFold


# public void build_models()

if __name__ == "__main__":
    """
    ### [24.05.27] Obs. >> NT(u)
    ## Opt. Rngs.
    top_n_tags_list = [10, 20, 30, 40, 50, 60]
    score_iter_list = [3]
    score_learing_rate_list = [0.0001]
    score_generalization_list = [0.001]
    weight_iter_list = [5]
    weight_learning_rate_list = [0.001]
    weight_generalization_list = [1.0]
    dv_score_list = [0.0]
    co_occur_items_list = [4]
    frob_norm_list = [1]
    """

    ### [24.05.29] Obs. >> NT(u) - SubMachine
    ## Opt. Rngs.
    top_n_tags_list = [10, 20, 30, 40, 50, 60]
    score_iter_list = [3]
    score_learing_rate_list = [0.01]
    score_generalization_list = [0.001]
    weight_iter_list = [5]
    weight_learning_rate_list = [0.001]
    weight_generalization_list = [1.0]
    dv_score_list = [0.0]
    co_occur_items_list = [4]
    frob_norm_list = [1]

    ## INVOKE
    # for selected_data in DataType:
    for selected_data in [DataType.E_COLLEY]:
        for top_n_tags in top_n_tags_list:
            for co_occur_items in co_occur_items_list:
                for score_iterations in score_iter_list:
                    for score_learning_rate in score_learing_rate_list:
                        for score_generalization in score_generalization_list:
                            for weight_iteration in weight_iter_list:
                                for weight_learning_rate in weight_learning_rate_list:
                                    for (
                                        weight_generalization
                                    ) in weight_generalization_list:
                                        for default_voting_score in dv_score_list:
                                            # run!
                                            build_models(
                                                selected_data=selected_data,
                                                default_voting_score=default_voting_score,
                                                top_n_tags=top_n_tags,
                                                frob_norm_n=1,
                                                co_occur_items=co_occur_items,
                                                score_iterations=score_iterations,
                                                score_learning_rate=score_learning_rate,
                                                score_generalization=score_generalization,
                                                weight_iterations=weight_iteration,
                                                weight_learning_rate=weight_learning_rate,
                                                weight_generalization=weight_generalization,
                                            )
                                        # end : for (DV_scores)
                                    # end : for (W_gamma)
                                # end : for (W_lambda)
                            # end : for (W_iter)
                        # end : for (S_gamma)
                    # end : for (S_lambda)
                # end : for (S_iter)
            # end : for (co_occur_items)
        # end : for (top_n_tags)
    # end : for (DataSet)
# end : main()
