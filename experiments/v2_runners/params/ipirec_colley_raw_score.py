"""
IPIRec - Heuristics approach
- Const Params: \\lambda = 0.1, \\gamma = 0.05, \\theta (I(x,y) = 5, NT(u) = 4)
- Adjust Params: iter = [40:200:40], FrobNorm = [1, 2], DV scores = [0.0, 0.1, 0.2],  \\theta (I(x,y) = [4:10:2]
"""

## build-in
import os
import sys

# [DEF] Environment variables
__dir_name__ = os.path.basename(os.path.dirname(__file__))
WORKSPACE_HOME = os.path.dirname(__file__).replace(f"/experiments/{__dir_name__}", "")
""".../ipirec"""
DATA_SET_HOME = f"{WORKSPACE_HOME}/data"
""">>> `${WORKSPACE_HOME}`/data"""
RESULTS_SUMMARY_HOME = f"{WORKSPACE_HOME}/results/240520/views"
""">>> `${WORKSPACE_HOME}`/results/YYMMDD/views"""
MODEL_NAME = "IPIRec"

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
        estimator_params = BiasedCorrelationEstimator.create_models_parameters(
            score_iterations=score_iterations,
            score_learning_rate=score_learning_rate,
            score_generalization=score_generalization,
            weight_iterations=weight_iterations,
            weight_learning_rate=weight_learning_rate,
            weight_generalization=weight_generalization,
            frob_norm=frob_norm_n,
            default_voting=default_voting_score,
        )

        estimator = BiasedCorrelationEstimator(
            model=model,
            model_params=estimator_params,
        )
        estimator.train(
            target_decision=DecisionType.E_VIEW,
            n=frob_norm_n,
            emit_iter_condition=weight_iterations,
        )
        """
        estimator.train(
            target_decision=DecisionType.E_LIKE,
            n=frob_norm_n,
            emit_iter_condition=weight_iterations,
        )
        estimator.train(
            target_decision=DecisionType.E_PURCHASE,
            n=frob_norm_n,
            emit_iter_condition=weight_iterations,
        )
        """
        recommender = ScoreBasedRecommender(
            estimator=estimator,
        )
        recommender.prediction()

        for target_decision in [
            DecisionType.E_VIEW,
            # DecisionType.E_LIKE,
            # DecisionType.E_PURCHASE,
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
            model_info_str += f"score_learing_rate: {score_learning_rate}\n"
            model_info_str += "[Estimator]\n"
            model_info_str += f"weight_learning_rate: {weight_learning_rate}\n"
            model_info_str += f"weight_generalization: {weight_generalization}\n"
            model_info_str += f"weight_iteration: {weight_iterations}\n"
            model_info_str += f"Frob_norm: {frob_norm_n}\n"
            model_info_str += f"default_voting: {default_voting_score}\n"

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
        """
        evaluator.evlautions_summary_df().to_csv(
            path_or_buf=results_summary_path,
            mode="at",
            encoding="utf-8",
        )
        """
    # end : KFold


# public void build_models()

if __name__ == "__main__":
    # [24.05.14] TEST
    """
    top_n_tags_list = [n for n in range(5, 25, 5)]
    score_iter_list = [10, 30]
    score_learing_rate_list = [0.01]
    score_generalization_list = [0.01]
    weight_learning_rate_list = [0.01]
    weight_generalization_list = [0.01]
    weight_iter_list = [10, 30]

    frob_norm_list = [1]
    dv_score_list = [0.0]
    co_occur_items_list = [5]
    """
    # [24.05.16] Observ.
    top_n_tags_list = [n for n in range(10, 25, 5)]
    score_iter_list = [10]
    score_learing_rate_list = [0.001]
    score_generalization_list = [0.1]
    weight_learning_rate_list = [0.001]
    weight_generalization_list = [0.1]
    weight_iter_list = [10]

    frob_norm_list = [1]
    dv_score_list = [0.0]
    co_occur_items_list = [4]

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
