"""
[작성일] 24.06.14 15:40
- 추천기를 달리했을 때의 추천결과 비교
"""

## build-in
import os
import sys
from datetime import datetime

import pickle
from pandas import DataFrame

# [DEF] Environment variables
__dir_name__ = os.path.basename(os.path.dirname(__file__))
__COMP_DATE_STR__ = datetime.now().strftime("%Y%m%d")
WORKSPACE_HOME = os.path.dirname(__file__).replace(f"experiments/{__dir_name__}", "")
""".../ipirec*"""
DATA_SET_HOME = f"{WORKSPACE_HOME}/data"
""">>> `${WORKSPACE_HOME}`/data"""
MODEL_NAME = "IBCF"
"""Obs."""
RESULTS_SUMMARY_HOME = f"{WORKSPACE_HOME}/results/{__COMP_DATE_STR__}"
""">>> `${WORKSPACE_HOME}`/results/`YYYYMMDD`"""

# [SET] Env.append($WORKSPACE_HOME)
sys.path.append(WORKSPACE_HOME)


# [IO] Binary instances paths variables
SUB_STORAGE_HOME = "/data/tghwang"
LOCAL_STORAGE_HOME = WORKSPACE_HOME
# BIN_INSTANCES_HOME = LOCAL_STORAGE_HOME
BIN_INSTANCES_HOME = SUB_STORAGE_HOME
RESOURCES_DIR_HOME = f"{BIN_INSTANCES_HOME}/resources"
""">>> `${BIN_INSTANCES_HOME}`/resources"""


## custom modules
from core import *
from colley import *
from ipirec import *
from itemcf import *


def build_models():
    data_name = "colley"
    KFold = 5
    top_n_conditions_list = [n for n in range(3, 37, 2)]
    dataset_dir_path = f"{WORKSPACE_HOME}/data/{data_name}"
    resource_dir_path = f"{RESOURCES_DIR_HOME}/{Pearson.__name__}"
    if not DirectoryPathValidator.exist_dir(resource_dir_path):
        DirectoryPathValidator.mkdir(resource_dir_path)

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
        # distance_file_path = os.path.dirname(train_file_path)

        estimator = AdjustedWeightedSum(model=model)
        recommender = ELABasedRecommender(
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
            model_info_str += f"Distance: {type(model).__name__}\n"
            model_info_str += "[Estimator]\n"
            model_info_str += f"Estimator: {type(estimator).__name__}\n"

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
        # end : KFold


# end : public void build_models()


if __name__ == "__main__":
    build_models()
# end : main()
