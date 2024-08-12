"""
[수정일] 24.07.03 17:48. 모듈재현하는 과정에서 실행오류가 있음 (코드 저장소가 꼬인듯...)
"""

## build-in
import os
import sys
import pickle

# 3rd Pty.
from pandas import DataFrame

# [DEF] Environment variables
__dir_name__ = os.path.basename(os.path.dirname(__file__))
# WORKSPACE_HOME = os.path.dirname(__file__).replace(f"/{__dir_name__}", "")
WORKSPACE_HOME = os.path.dirname(__file__).replace(f"/experiments/{__dir_name__}", "")
""".../ipirec"""
DATA_SET_HOME = f"{WORKSPACE_HOME}/data"
""">>> `${WORKSPACE_HOME}`/data"""
RESULTS_SUMMARY_HOME = f"{WORKSPACE_HOME}/results/redef"
""">>> `${WORKSPACE_HOME}`/results/redef"""
MODEL_NAME = "IPIRec_v1"

# [IO] Binary instances paths variables
SUB_STORAGE_HOME = "/data/tghwang"
LOCAL_STORAGE_HOME = WORKSPACE_HOME
BIN_INSTANCES_HOME = LOCAL_STORAGE_HOME
# BIN_INSTANCES_HOME = SUB_STORAGE_HOME
RESOURCES_DIR_HOME = f"{BIN_INSTANCES_HOME}/resources/{MODEL_NAME}"
""">>> `${WORKSPACE_HOME}`/resources`"""

# [SET] Env.append($WORKSPACE_HOME)
sys.path.append(WORKSPACE_HOME)


## custom modules
from core import *
from colley import *
from ipirec import *
from movielens import *


if __name__ == "__main__":
    ## Vars. Def.
    FOLD_SET_ID = int(sys.argv[1]) if len(sys.argv) == 2 else 0
    selected_dataset = DataType.E_COLLEY
    target_decision = DecisionType.E_PURCHASE
    top_n_tags = 10
    co_occur_items = 5
    score_iter = 10
    score_learning_rate = 0.01
    score_generalization = 0.01
    weight_iter = 10
    weight_learning_rate = 0.01
    weight_generalization = 0.01
    default_voting = 0.0
    frob_norm = 1
    top_n_items = [n for n in range(3, 37, 2)]

    ## [IO] Dataset
    dataset_home_path = str.format(
        "{0}/{1}",
        DATA_SET_HOME,
        DataType.to_str(selected_dataset),
    )
    dataset: BaseDataSet = None
    dump_dir_path = str.format(
        "{0}/{1}/{2}",
        RESOURCES_DIR_HOME,
        MODEL_NAME,
        DataType.to_str(selected_dataset),
    )
    dump_model_file_path = str.format(
        "{0}/{1}.bin",
        dump_dir_path,
        FOLD_SET_ID,
    )
    test_file_path = str.format(
        "{0}/{1}_{2}_{3}_list.csv",
        dataset_home_path,
        "test",
        FOLD_SET_ID,
        DecisionType.to_str(target_decision),
    )
    if not DirectoryPathValidator.exist_dir(dump_dir_path):
        DirectoryPathValidator.mkdir(dump_dir_path)

    if os.path.exists(dump_model_file_path):
        with open(file=dump_model_file_path, mode="rb") as fin:
            match (selected_dataset):
                case DataType.E_COLLEY:
                    dataset: ColleyFilteredDataSet = pickle.load(fin)
                case DataType.E_MOVIELENS:
                    dataset: MovieLensFilteredDataSet = pickle.load(fin)
                case _:
                    fin.close()
                    raise ValueError()
            fin.close()
        # end : StreamReader()
    else:
        # metadata;
        match (selected_dataset):
            case DataType.E_COLLEY:
                dataset = ColleyFilteredDataSet(
                    dataset_dir_path=dataset_home_path,
                )
            case DataType.E_MOVIELENS:
                dataset = MovieLensFilteredDataSet(
                    dataset_dir_path=dataset_home_path,
                )
            case _:
                raise ValueError()

        # decisions
        dataset._load_metadata_()
        training_file_path = ""
        for decision_type in DecisionType:
            training_file_path = str.format(
                "{0}/{1}_{2}_{3}_list.csv",
                dataset_home_path,
                "train",
                FOLD_SET_ID,
                DecisionType.to_str(decision_type),
            )
            dataset.append_decisions(
                file_path=training_file_path,
                decision_type=decision_type,
            )
        # end : for (decision_types)
        dataset.__id_index_mapping__()

        # model dump
        with open(file=dump_model_file_path, mode="wb") as fout:
            pickle.dump(dataset, fout)
            fout.close()
    # end : if (DataSet)

    ## Build model
    # datamodel
    model_params = CorrelationModel.create_models_parameters(
        top_n_tags=top_n_tags,
        co_occur_items_threshold=co_occur_items,
    )
    model = CorrelationModel(
        dataset=dataset,
        model_params=model_params,
    )
    model.analysis()

    # estimator
    model_params = BiasedCorrelationEstimator.create_models_parameters(
        score_iterations=score_iter,
        score_learning_rate=score_learning_rate,
        score_generalization=score_generalization,
        weight_iterations=weight_iter,
        weight_learning_rate=weight_learning_rate,
        weight_generalization=weight_generalization,
        frob_norm=frob_norm,
        default_voting=default_voting,
    )
    estimator = BiasedCorrelationEstimator(
        model=model,
        model_params=model_params,
    )
    for decision_type in DecisionType:
        estimator.train(
            target_decision=decision_type,
            n=frob_norm,
            emit_iter_condition=weight_iter,
        )
    # end : for (decisions)

    # recommender
    recommender = ScoreBasedRecommender(
        estimator=estimator,
    )
    recommender.prediction()

    # evaluator
    evaluator = IRMetricsEvaluator(
        recommender=recommender,
        file_path=test_file_path,
    )
    evaluator.top_n_eval(top_n_conditions=top_n_items)
    df: DataFrame = evaluator.evlautions_summary_df()

    print(df)
# end : main()
