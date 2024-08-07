"""
[작성일] 24.06.14 14:17
- 추천기를 달리했을 때의 추천결과 비교
"""

## build-in
import os
import sys
from datetime import datetime

from pandas import DataFrame

# [DEF] Environment variables
__dir_name__ = os.path.basename(os.path.dirname(__file__))
__COMP_DATE_STR__ = datetime.now().strftime("%Y%m%d")
WORKSPACE_HOME = os.path.dirname(__file__).replace(f"experiments/{__dir_name__}", "")
""".../ipirec*"""
DATA_SET_HOME = f"{WORKSPACE_HOME}/data"
""">>> `${WORKSPACE_HOME}`/data"""
MODEL_NAME = "NMF"
"""Obs."""
RESULTS_SUMMARY_HOME = f"{WORKSPACE_HOME}/results/{__COMP_DATE_STR__}"
""">>> `${WORKSPACE_HOME}`/results/`YYYYMMDD`"""

# [SET] Env.append($WORKSPACE_HOME)
sys.path.append(WORKSPACE_HOME)

## custom modules
from core import *
from colley import *
from ipirec import *
from decompositions import NMFDecompositionModel, DecompositionsEstimator


def build_models(
    factors_dim: int,
    factors_iter: int,
    train_iter: int,
    learning_rate: float,
    generalization: float,
    frob_norm_n: int,
    decision_type_seq_list: list = [
        DecisionType.E_VIEW,
        DecisionType.E_PURCHASE,
    ],
):
    data_name = "colley"
    dataset_dir_path = f"{DATA_SET_HOME}/{data_name}"
    KFold = 5
    top_n_conditions_list = [n for n in range(3, 37, 2)]

    for k in range(KFold):
        ## [DEF] models
        model_info_str = ""
        dataset = ColleyFilteredDataSet(dataset_dir_path)
        model: BaseModel = None
        estimator: BaseEstimator = None
        recommender: BaseRecommender = None
        evaluator: BaseEvaluator = None
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

        for decision_type in decision_type_seq_list:
            estimator.train(
                target_decision=decision_type,
                n=frob_norm_n,
                emit_iter_condition=train_iter,
            )
        # end : for (decision_types)

        # recommender = ELABasedRecommender(estimator=estimator)
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
            model_info_str += f"dimensions: {factors_dim}\n"
            model_info_str += f"factorizer_iters: {factors_iter}\n"
            model_info_str += "[Estimator]\n"
            model_info_str += f"learning_rate: {learning_rate}\n"
            model_info_str += f"generalization: {generalization}\n"
            model_info_str += f"train_iterations: {train_iter}\n"
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
            df: DataFrame = evaluator.save_evaluations_summary(
                file_path=results_summary_path,
                mode="at",
            )
            print(df)
        # end : for (positive_decision_types)
    # end : KFold


# public void build_models()


if __name__ == "__main__":

    ### Fixed Mod. Opt. ###
    factors_dim_list = [200]
    factorizer_iters_list = [200]
    learning_rate_list = [10 ** (-1 * 4)]
    generalizations_list = [10 ** (-1 * 5)]
    train_iters_list = [150]

    ## Train. OPT.
    frob_norm_list = [1]
    decision_type_seq_list = [
        DecisionType.E_VIEW,
        DecisionType.E_LIKE,
        DecisionType.E_PURCHASE,
    ]

    ## INVOKE
    for frob_norm_n in frob_norm_list:
        for factors_dim in factors_dim_list:
            for factors_iter in factorizer_iters_list:
                for learning_rate in learning_rate_list:
                    for generalization in generalizations_list:
                        for train_iters in train_iters_list:
                            build_models(
                                factors_dim=factors_dim,
                                factors_iter=factors_iter,
                                train_iter=train_iters,
                                learning_rate=learning_rate,
                                generalization=generalization,
                                frob_norm_n=frob_norm_n,
                                decision_type_seq_list=decision_type_seq_list,
                            )
                        # end : for (Iters)
                    # end : for (generalizations)
                # end : for (learning_rates)
            # end : for (IoFs)
        # end : for (DoFs)
    # end : for (Ln-Norms)
# end : main()
