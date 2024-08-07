## build-in
import os
import sys

# [DEF] Environment variables
__dir_name__ = os.path.basename(os.path.dirname(__file__))
WORKSPACE_HOME = os.path.dirname(__file__).replace(f"/experiments/{__dir_name__}", "")
""".../ipirec"""
DATA_SET_HOME = f"{WORKSPACE_HOME}/data"
""">>> `${WORKSPACE_HOME}`/data"""
RESULTS_SUMMARY_HOME = f"{WORKSPACE_HOME}/results/params"
""">>> `${WORKSPACE_HOME}`/results/params"""
MODEL_NAME = "IPIRec"

# [SET] Env.append($WORKSPACE_HOME)
sys.path.append(WORKSPACE_HOME)

## custom modules
from core import *
from lc_corr import *


def build_models(
    selected_data: DataType,
    default_voting_score: float,
    top_n_tags: int,
    frob_norm_n: int,
    co_occur_items: int,
    score_learning_rate: float,
    score_iteration: int,
    weight_iterations: int,
    weight_learning_rate: float,
    weight_generalization: float,
):
    data_name = DataType.to_str(selected_data)
    KFold = 5
    top_n_conditions_list = [n for n in range(3, 37, 2)]
    dataset_dir_path = f"{DATA_SET_HOME}/{data_name}"
    # results_summary_path = f"{WORKSPACE_HOME}/results/params/IPIRec/{data_name}_TopN_{decision_kwd}_IRMetrics_{DirectoryPathValidator.current_datetime_str()}.csv"

    """
    results_summary_path = str.format(
        "{0}/{1}/{2}_TopN_IRMetrics_{3}.csv",
        # "{0}/{1}/{2}_TopN_{3}_IRMetrics_{4}.csv",
        RESULTS_SUMMARY_HOME,
        MODEL_NAME,
        data_name,
        # decision_kwd,
        DirectoryPathValidator.current_datetime_str(),
    )
    __dir_path_str = os.path.dirname(results_summary_path)
    if not DirectoryPathValidator.exist_dir(__dir_path_str):
        DirectoryPathValidator.mkdir(__dir_path_str)
    """

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
            iterations_threshold=score_iteration,
            learning_rate=score_learning_rate,
        )
        """
        with open(
            file=results_summary_path,
            mode="at",
            encoding="utf-8",
        ) as fout:
            fout.write("[DataSet]\n")
            fout.write(f"Target,{data_name},{decision_kwd}\n")
            fout.write(f"No.,{k + 1},{KFold}\n")
            fout.write("[Model]\n")
            fout.write(f"top_n_tag: {top_n_tags}\n")
            fout.write(f"co_occur_items: {co_occur_items}\n")
            fout.write(f"score_iterations: {score_iteration}\n")
            fout.write(f"score_learing_rate: {score_learning_rate}\n")
            fout.close()
        # end : StreamWriter (ModelOpt.)
        """
        model = CorrelationModel(
            dataset=dataset,
            model_params=model_params,
        )
        model.analysis()
        estimator_params = BiasedEstimator.create_models_parameters(
            default_voting_score=default_voting_score,
            learning_rate=weight_learning_rate,
            generalization=weight_generalization,
        )
        """
        with open(
            file=results_summary_path,
            mode="at",
            encoding="utf-8",
        ) as fout:
            fout.write("[Estimator]\n")
            fout.write(f"weight_learning_rate: {weight_learning_rate}\n")
            fout.write(f"weight_generalization: {weight_generalization}\n")
            fout.write(f"weight_iteration: {weight_iterations}\n")
            fout.write(f"Frob_norm: {frob_norm_n}\n")
            fout.close()
        # end : StreamWriter (EstimatorOpt.)
        """

        estimator = BiasedEstimator(
            model=model,
            model_params=estimator_params,
        )
        estimator.train(
            target_decision=DecisionType.E_VIEW,
            n=frob_norm_n,
            emit_iter_condition=weight_iterations,
        )
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
        recommender = ScoreBasedRecommender(
            estimator=estimator,
        )
        recommender.prediction()

        for target_decision in [
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
            model_info_str += f"score_iterations: {score_iteration}\n"
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
    ## MODEL PARAMS
    default_voting_score = 0.1
    top_n_tags_list: list = [n for n in range(5, 35, 5)]
    co_occur_items_list: list = [n for n in range(5, 45, 5)]
    score_learning_rate_list = [0.1]
    """
    score_learning_rate_list = sorted(
        [
            round(
                number=r * (10**-n),
                ndigits=n + 1,
            )
            for r in [0.5, 0.1]
            for n in range(0, 5, 1)
        ],
        reverse=True,
    )
    """
    score_iterations_list = [40]  # [n for n in range(10, 40, 10)]
    weight_iterations_list = [30]  # [n for n in range(30, 90, 10)]
    weight_learning_rate_list = [0.01]
    """
    weight_learning_rate_list = sorted(
        [
            round(
                number=r * (10**-n),
                ndigits=n + 1,
            )
            for r in [0.5, 0.1]
            for n in range(4, 8, 1)
        ],
        reverse=True,
    )
    """
    weight_generalizations_list = [0.01]  # [g * 0.1 for g in weight_learning_rate_list]

    ## INVOKE
    # for selected_data in DataType:
    for selected_data in [DataType.E_MOVIELENS]:
        for top_n_tags in top_n_tags_list:
            for co_occur_items in co_occur_items_list:
                for score_learning_rate in score_learning_rate_list:
                    for score_iteration in score_iterations_list:
                        for weight_iteration in weight_iterations_list:
                            for weight_learning_rate in weight_learning_rate_list:
                                for (
                                    weight_generalization
                                ) in weight_generalizations_list:
                                    # run!
                                    build_models(
                                        selected_data=selected_data,
                                        default_voting_score=default_voting_score,
                                        top_n_tags=top_n_tags,
                                        frob_norm_n=1,
                                        co_occur_items=co_occur_items,
                                        score_learning_rate=score_learning_rate,
                                        score_iteration=score_iteration,
                                        weight_iterations=weight_iteration,
                                        weight_learning_rate=weight_learning_rate,
                                        weight_generalization=weight_generalization,
                                    )
                                # end : for (W_gamma)
                            # end : for (W_lambda)
                        # end : for (W_iter)
                    # end : for (S_iter)
                # end : for (S_lambda)
            # end : for (co_occur_items)
        # end : for (top_n_tags)
    # end : for (DataSet)
# end : main()
