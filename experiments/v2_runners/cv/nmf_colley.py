import os
import sys

WORKSPACE_HOME = os.path.dirname(__file__).replace("/experiments/cv", "")
sys.path.append(WORKSPACE_HOME)

from core import *
from decompositions import *
from lc_corr import *

if __name__ == "__main__":
    data_name = DataType.to_str(DataType.E_COLLEY)
    dataset_dir_path = f"{WORKSPACE_HOME}/data/{data_name}"

    KFold = 5
    top_n_conditions_list = [n for n in range(3, 37, 2)]
    for k in range(KFold):
        dataset = ColleyFilteredDataSet(dataset_dir_path)
        dataset._load_metadata_()
        for selected_decision in DecisionType:
            kwd = DecisionType.to_str(selected_decision)
            train_file_path = f"{dataset_dir_path}/train_{k}_{kwd}_list.csv"
            dataset.append_decisions(
                train_file_path,
                selected_decision,
            )
        dataset.__id_index_mapping__()

        ## [IO] model
        model_params = NMFDecompositionModel.create_models_parameters(
            factors_dim=300,
            factorizer_iters=50,
        )
        model = NMFDecompositionModel(dataset=dataset, model_params=model_params)
        model.analysis()

        model_params = DecompositionsEstimator.create_models_parameters(
            learning_rate=0.00001,
            generalization=0.000001,
        )
        estimator = DecompositionsEstimator(model=model, model_params=model_params)
        estimator.train(
            DecisionType.E_VIEW,
            n=1,
            emit_iter_condition=40,
        )
        estimator.train(
            DecisionType.E_LIKE,
            n=1,
            emit_iter_condition=40,
        )
        estimator.train(
            DecisionType.E_PURCHASE,
            n=1,
            emit_iter_condition=40,
        )
        recommender = ScoreBasedRecommender(
            estimator=estimator,
        )
        recommender.prediction()

        # target: LIKES
        selected_test_decision = DecisionType.E_LIKE
        kwd_str = DecisionType.to_str(selected_test_decision)
        results_summary_path = (
            f"{WORKSPACE_HOME}/results/cv/NMF/{data_name}_TopN_{kwd_str}_IRMetrics.csv"
        )
        test_file_path = f"{dataset_dir_path}/test_{k}_{kwd_str}_list.csv"

        eval = IRMetricsEvaluator(
            recommender=recommender,
            file_path=test_file_path,
        )
        eval.top_n_eval(top_n_conditions=top_n_conditions_list)
        eval.evlautions_summary_df().to_csv(
            path_or_buf=results_summary_path,
            mode="at",
            encoding="utf-8",
        )

        # target: PURCHASES
        selected_test_decision = DecisionType.E_PURCHASE
        kwd_str = DecisionType.to_str(selected_test_decision)
        results_summary_path = (
            f"{WORKSPACE_HOME}/results/cv/NMF/{data_name}_TopN_{kwd_str}_IRMetrics.csv"
        )
        test_file_path = f"{dataset_dir_path}/test_{k}_{kwd_str}_list.csv"

        eval = IRMetricsEvaluator(
            recommender=recommender,
            file_path=test_file_path,
        )
        eval.top_n_eval(top_n_conditions=top_n_conditions_list)
        eval.evlautions_summary_df().to_csv(
            path_or_buf=results_summary_path,
            mode="at",
            encoding="utf-8",
        )
    # end : for (KFoldCV)
# end : main()
