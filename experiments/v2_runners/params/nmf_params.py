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
MODEL_NAME = "NMF"

# [SET] Env.append($WORKSPACE_HOME)
sys.path.append(WORKSPACE_HOME)

## custom modules
from core import *
from lc_corr import *
from decompositions import *


def build_models(
    selected_data: DataType,
    factors_dim: int,
    factors_iter: int,
    frob_norm_n: int,
    learning_rate: float,
    generalization: float,
    train_iter: int,
) -> None:
    KFold = 5
    top_n_conditions_list = [n for n in range(3, 37, 2)]
    data_name = DataType.to_str(selected_data)
    dataset_dir_path = f"{DATA_SET_HOME}/{data_name}"

    for k in range(KFold):
        ## [DEF] models
        model_info_str = ""
        dataset: BaseDataSet = None
        model: DecompositionModel = None
        estimator: DecompositionsEstimator = None
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

        model_params = NMFDecompositionModel.create_models_parameters(
            factors_dim=factors_dim,
            factorizer_iters=factors_iter,
        )
        model = NMFDecompositionModel(
            dataset=dataset,
            model_params=model_params,
        )
        model.analysis()
        estimator_params = DecompositionsEstimator.create_models_parameters(
            learning_rate=learning_rate,
            generalization=generalization,
        )
        estimator = DecompositionsEstimator(
            model=model,
            model_params=estimator_params,
        )
        for target_decision in DecisionType:
            estimator.train(
                target_decision=target_decision,
                n=frob_norm_n,
                emit_iter_condition=train_iter,
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
                "{0}/{1}/{2}_TopN_{3}_IRMetrics_{4}.csv",
                RESULTS_SUMMARY_HOME,
                MODEL_NAME,
                data_name,
                decision_kwd,
                DirectoryPathValidator.current_datetime_str(),
            )
            __dir_path_str = os.path.dirname(results_summary_path)
            if not DirectoryPathValidator.exist_dir(__dir_path_str):
                DirectoryPathValidator.mkdir(__dir_path_str)

            test_file_path = f"{dataset_dir_path}/test_{k}_{decision_kwd}_list.csv"
            evaluator = IRMetricsEvaluator(
                recommender=recommender,
                file_path=test_file_path,
            )

            model_info_str = ""
            model_info_str += "[DataSet]\n"
            model_info_str += f"Target,{data_name},{decision_kwd}\n"
            model_info_str += f"No.,{k + 1},{KFold}\n"
            model_info_str += "[Model]\n"
            model_info_str += f"factors, {factors_dim}\n"
            model_info_str += f"factors iter, {factors_iter}\n"
            model_info_str += "[Estimator]\n"
            model_info_str += f"weight_learning_rate: {learning_rate}\n"
            model_info_str += f"weight_generalization: {generalization}\n"
            model_info_str += f"weight_iteration: {train_iter}\n"
            model_info_str += f"Frob_norm: {frob_norm_n}\n"

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
        # end : for (test-set = [Likes, Purchases])
    # end : for (KFold)


# end : build_models

if __name__ == "__main__":
    frob_norm_n = 1
    factors_dim_list = [200, 250, 300]
    factors_iter_list = [iter for iter in range(200, 400, 50)]
    learning_rate_list = [0.001, 0.0001, 0.00001, 0.000001]
    generalizations_list = [g * 0.1 for g in learning_rate_list]
    train_iterations_list = [30, 50, 100, 150]

    for selected_data in DataType:
        for factors_dim in factors_dim_list:
            for factors_iter in factors_iter_list:
                for learning_rate in learning_rate_list:
                    for generalization in generalizations_list:
                        for train_iter in train_iterations_list:
                            build_models(
                                selected_data=selected_data,
                                factors_dim=factors_dim,
                                factors_iter=factors_iter,
                                frob_norm_n=frob_norm_n,
                                learning_rate=learning_rate,
                                generalization=generalization,
                                train_iter=train_iter,
                            )
                        # end : for (train_iter)
                    # end : for (generalization)
                # end : for (learning_rates)
            # end : for (factorizer_iter)
        # end : for (factors_dim)
    # end : for (dataset = [ML, Colley])
# end : main()
