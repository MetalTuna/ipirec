import os
import sys

WORKSPACE_HOME = os.path.dirname(__file__).replace("/experiments/cv", "")
sys.path.append(WORKSPACE_HOME)

import pickle

from core import *
from colley import *
from movielens import *
from itemcf import *
from lc_corr import *

if __name__ == "__main__":
    data_name = DataType.to_str(DataType.E_MOVIELENS)

    dataset_dir_path = f"{WORKSPACE_HOME}/data/{data_name}"
    resource_dir_path = f"{WORKSPACE_HOME}/resources/similarities"
    DirectoryPathValidator.mkdir(resource_dir_path)

    KFold = 5
    top_n_conditions_list = [n for n in range(3, 37, 2)]
    for k in range(KFold):
        dataset = MovieLensFilteredDataSet(dataset_dir_path)
        dataset._load_metadata_()
        for selected_decision in DecisionType:
            kwd = DecisionType.to_str(selected_decision)
            train_file_path = f"{dataset_dir_path}/train_{k}_{kwd}_list.csv"
            dataset.append_decisions(
                train_file_path,
                selected_decision,
            )
        dataset.__id_index_mapping__()
        similarity_file_path = f"{resource_dir_path}/{data_name}_{k}_item_pcc.bin"

        ## [IO] model
        model: BaseDistanceModel = None
        if os.path.exists(similarity_file_path):
            with open(file=similarity_file_path, mode="rb") as fin:
                model: Pearson = pickle.load(fin)
                fin.close()
        else:
            model = Pearson(dataset=dataset)
            model.analysis()
            with open(
                file=similarity_file_path,
                mode="wb",
            ) as fout:
                pickle.dump(model, fout)
                fout.close()

        distance_file_path = os.path.dirname(train_file_path)

        estimator = AdjustedWeightedSum(model=model)
        recommender = ScoreBasedRecommender(
            estimator=estimator,
        )
        recommender.prediction()

        # target: Views
        selected_test_decision = DecisionType.E_VIEW
        kwd_str = DecisionType.to_str(selected_test_decision)
        results_summary_path = f"{WORKSPACE_HOME}/results/cv/ItemCF/{data_name}_TopN_{kwd_str}_IRMetrics.csv"
        test_file_path = f"{dataset_dir_path}/test_{k}_{kwd_str}_list.csv"
        res_dir_path = os.path.dirname(results_summary_path)
        if not DirectoryPathValidator.exist_dir(res_dir_path):
            DirectoryPathValidator.mkdir(res_dir_path)

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

        """
        # target: LIKES
        selected_test_decision = DecisionType.E_LIKE
        kwd_str = DecisionType.to_str(selected_test_decision)
        results_summary_path = f"{WORKSPACE_HOME}/results/cv/ItemCF/{data_name}_TopN_{kwd_str}_IRMetrics.csv"
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
        results_summary_path = f"{WORKSPACE_HOME}/results/cv/ItemCF/{data_name}_TopN_{kwd_str}_IRMetrics.csv"
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
        """
    # end : for (KFoldCV)
# end : main()
