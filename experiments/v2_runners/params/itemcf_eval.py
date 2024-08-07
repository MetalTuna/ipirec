## build-in
import os
import sys

import pickle

# [DEF] Environment variables
__dir_name__ = os.path.basename(os.path.dirname(__file__))
WORKSPACE_HOME = os.path.dirname(__file__).replace(f"/experiments/{__dir_name__}", "")
""".../ipirec"""
DATA_SET_HOME = f"{WORKSPACE_HOME}/data"
""">>> `${WORKSPACE_HOME}`/data"""
RESULTS_SUMMARY_HOME = f"{WORKSPACE_HOME}/results/params"
""">>> `${WORKSPACE_HOME}`/results/params"""
DISTANCE_DUMP_HOME = f"{WORKSPACE_HOME}/resources/pearson"
""">>> `${WORKSPACE_HOME}`/resources/pearson"""
MODEL_NAME = "ItemCF"

# [SET] Env.append($WORKSPACE_HOME)
sys.path.append(WORKSPACE_HOME)

## custom modules
from core import *
from itemcf import *
from lc_corr import *


def build_models(
    selected_data: DataType,
    selected_decision: DecisionType,
):
    data_name = DataType.to_str(selected_data)
    decision_kwd = DecisionType.to_str(selected_decision)
    KFold = 5
    top_n_conditions_list = [n for n in range(3, 37, 2)]
    dataset_dir_path = f"{DATA_SET_HOME}/{data_name}"
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

    for k in range(KFold):
        ## [DEF] models
        dataset: BaseDataSet = None
        model: BaseDistanceModel = None
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

        # [Analysis] Similarities
        __dir_path_str = f"{DISTANCE_DUMP_HOME}/cv"
        file_path = f"{__dir_path_str}/{data_name}_{k}_item_pcc.bin"
        if os.path.exists(file_path):
            with open(file=file_path, mode="rb") as fin:
                model: Pearson = pickle.load(fin)
                fin.close()
            # end : StreamReader()
        else:
            DirectoryPathValidator.mkdir(__dir_path_str)
            model = Pearson(dataset=dataset)
            model.analysis()
            with open(file=file_path, mode="wb") as fout:
                pickle.dump(model, file=fout)
                fout.close()
            # end : StreamWriter()

        # [Prediction]
        estimator = AdjustedWeightedSum(model=model)
        recommender = ScoreBasedRecommender(estimator=estimator)
        recommender.prediction()
        with open(
            file=results_summary_path,
            mode="at",
            encoding="utf-8",
        ) as fout:
            fout.write("[DataSet]\n")
            fout.write(f"Target,{data_name},{decision_kwd}\n")
            fout.write(f"No.,{k + 1},{KFold}\n")
            fout.write("[Model]\n")
            fout.write("Pearson\n")
            fout.write("[Estimator]\n")
            fout.write("AdjustedWeightedSum\n")
            fout.close()
        # end : StreamWriter (EstimatorOpt.)

        test_file_path = f"{dataset_dir_path}/test_{k}_{decision_kwd}_list.csv"
        evaluator = IRMetricsEvaluator(
            recommender=recommender,
            file_path=test_file_path,
        )
        evaluator.top_n_eval(top_n_conditions=top_n_conditions_list)
        evaluator.save_evaluations_summary(
            file_path=results_summary_path,
        )


# end : public build_models()

if __name__ == "__main__":
    for selected_data in DataType:
        for selected_decision in DecisionType:
            build_models(selected_data, selected_decision)
        # end : for (TargetDecisions = [Like, Purchase])
    # end : for (DataSet = [MovieLens, Colley])
# end : main()
