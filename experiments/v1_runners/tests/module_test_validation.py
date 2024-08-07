import os
import sys

WORKSPACE_HOME = os.path.dirname(__file__).replace("/experiments", "")
sys.path.append(WORKSPACE_HOME)
# print(WORKSPACE_HOME)
# exit()

from core import *
from lc_corr import *

if __name__ == "__main__":
    dataset_dir_path = f"{WORKSPACE_HOME}/data/colley"
    results_summary_path = (
        f"{WORKSPACE_HOME}/results/cv/IPIRec/Colley_TopN_IRMetrics.csv"
    )
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
        test_file_path = f"{dataset_dir_path}/test_{k}_purchase_list.csv"

        model_params = CorrelationModel.create_models_parameters(
            top_n_tags=30,
            co_occur_items_threshold=5,
            iterations_threshold=40,
            learning_rate=0.01,
        )
        model = CorrelationModel(
            dataset=dataset,
            model_params=model_params,
        )
        model.analysis()

        model_params = BiasedEstimator.create_models_parameters(
            learning_rate=0.01,
            generalization=0.001,
        )
        estimator = BiasedEstimator(
            model=model,
            model_params=model_params,
        )
        estimator.train(
            DecisionType.E_VIEW,
            n=1,
            emit_diff_condition=30,
        )
        estimator.train(
            DecisionType.E_LIKE,
            n=1,
            emit_diff_condition=30,
        )
        estimator.train(
            DecisionType.E_PURCHASE,
            n=1,
            emit_diff_condition=30,
        )

        recommender = ScoreBasedRecommender(
            estimator=estimator,
        )
        recommender.prediction()
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
