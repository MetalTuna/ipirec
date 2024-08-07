# build-in
import os
import pickle
import sys

# 3rd Pty. LIB.
from pandas import DataFrame

### [BEGIN]
# Environment variables
__dir_name__ = os.path.basename(os.path.dirname(__file__))
WORKSPACE_HOME = os.path.dirname(__file__).replace(f"/runners/{__dir_name__}", "")
DATA_SET_HOME = f"{WORKSPACE_HOME}/data/colley"
""">>> `${WORKSPACE_HOME}`/data"""
MODEL_NAME = "IBCF"

# custom modules
sys.path.append(WORKSPACE_HOME)
from core import *
from colley import *
from decompositions import *
from itemcf import *
from ipirec import *

# [IO] Binary instances paths variables
LOCAL_STORAGE_HOME = WORKSPACE_HOME
BIN_INSTANCES_HOME = LOCAL_STORAGE_HOME
# BIN_INSTANCES_HOME = SUB_STORAGE_HOME
DATE_STR = DirectoryPathValidator.current_datetime_str().split("_")[0].strip()
"""YYYYMMDD"""
TIME_STR = DirectoryPathValidator.current_datetime_str().split("_")[1].strip()
"""HHMMSS"""
RESOURCES_DIR_HOME = f"{BIN_INSTANCES_HOME}/resources/{DATE_STR}/{MODEL_NAME}"
""">>> `${BIN_INSTANCES_HOME}`/resources/${MODEL_NAME}"""
RESULTS_SUMMARY_HOME = f"{WORKSPACE_HOME}/results/{DATE_STR}/{MODEL_NAME}"
""">>> `${WORKSPACE_HOME}`/results/${YYYYMMDD}/${MODEL_NAME}"""
### [EMIT]


def main() -> None:
    _K_FOLD = 5
    _positive_decision_types = [
        DecisionType.E_LIKE,
        DecisionType.E_PURCHASE,
    ]
    _top_n_conditions = [n for n in range(3, 37, 2)]
    metrics_df_dict = dict()

    for k in range(_K_FOLD):
        dataset: ColleyDataSet = build_dataset(k)
        model: Pearson = build_model(dataset, k)
        estimator: AdjustedWeightedSum = build_estimator(model, k)
        recommender: ScoreBasedRecommender = score_recommender(estimator)
        recommender.prediction()
        metrics_df_dict.update({k: dict()})

        for decision_type in _positive_decision_types:
            _test_set_file_path = dataset.kfold_file_path(k, decision_type, False)
            evaluator = IRMetricsEvaluator(recommender, _test_set_file_path)
            evaluator.top_n_eval(_top_n_conditions)
            retrieved_df = evaluator.evlautions_summary_df()
            # metrics_df_dict[k].update({decision_type: retrieved_df})
            tags_freq_eval = CosineItemsTagsFreqAddPenalty()
            score_eval = TagsScoreRMSEEvaluator(recommender, _test_set_file_path)
            score_eval.eval()
            tags_dist_score: float = tags_freq_eval.tags_freq_distance(
                _test_set_file_path,
                recommender,
            )
            summary_results()
        # end : for (positive_decisions)
    # end : for (KFold)


# end : main()


def __append_results__(
    kfold_no: int,
    decision_type: DecisionType,
    retrieved_df: DataFrame,
    tags_dist_score: float,
    score_eval: TagsScoreRMSEEvaluator,
    model_params: dict = None,
    estimator_params: dict = None,
) -> None:
    raise NotImplementedError()


def summary_results():
    pass


def build_dataset(
    fold_set_no: int,
) -> ColleyDataSet:
    _dataset: ColleyDataSet = None
    _BIN_DATASET_FILE_PATH = str.format(
        "{0}/{1}/{2}.bin",
        RESOURCES_DIR_HOME,
        ColleyDataSet.__name__,
        fold_set_no,
    )

    if os.path.exists(_BIN_DATASET_FILE_PATH):
        with open(file=_BIN_DATASET_FILE_PATH, mode="rb") as fin:
            _dataset: ColleyDataSet = pickle.load(fin)
            fin.close()
        # end : StreamReader()
    else:
        _dataset = ColleyDataSet(
            dataset_dir_path=DATA_SET_HOME,
        )
        _dataset.load_kfold_train_set(
            kfold_set_no=fold_set_no,
        )
        _dataset.append_interest_tags()
        _BIN_DATASET_DIR_PATH = os.path.dirname(_BIN_DATASET_FILE_PATH)
        if not DirectoryPathValidator.exist_dir(_BIN_DATASET_DIR_PATH):
            DirectoryPathValidator.mkdir(_BIN_DATASET_DIR_PATH)
        with open(
            file=_BIN_DATASET_FILE_PATH,
            mode="wb",
        ) as fout:
            pickle.dump(
                _dataset,
                fout,
            )
            fout.close()
        # end : StreamWriter()
    # end : if (EXIST_BIN_DATASET)
    return _dataset


# end : build_dataset()


def build_model(
    dataset: ColleyDataSet,
    fold_set_no: int,
) -> Pearson:
    _model: Pearson = None
    if dataset == None:
        dataset = build_dataset(
            fold_set_no=fold_set_no,
        )
    _BIN_MODEL_FILE_PATH = str.format(
        "{0}/{1}/{2}.bin",
        RESOURCES_DIR_HOME,
        Pearson.__name__,
        fold_set_no,
    )
    if os.path.exists(_BIN_MODEL_FILE_PATH):
        with open(file=_BIN_MODEL_FILE_PATH, mode="rb") as fin:
            _model: Pearson = pickle.load(fin)
            fin.close()
        # end : StreamReader()
    else:
        _model = Pearson(dataset=dataset)
        _model.analysis()
        _BIN_MODEL_DIR_PATH = os.path.dirname(_BIN_MODEL_FILE_PATH)
        if not DirectoryPathValidator.exist_dir(_BIN_MODEL_DIR_PATH):
            DirectoryPathValidator.mkdir(_BIN_MODEL_DIR_PATH)
        with open(file=_BIN_MODEL_FILE_PATH, mode="wb") as fout:
            pickle.dump(_model, fout)
            fout.close()
        # end : StreamWriter()
    # end : if (EXIST_BIN_MODEL)

    return _model


# end : build_model()


def build_estimator(
    model: Pearson,
    fold_set_no: int,
) -> AdjustedWeightedSum:
    _estimator: AdjustedWeightedSum = None
    _BIN_MODEL_FILE_PATH = str.format(
        "{0}/{1}/{2}.bin",
        RESOURCES_DIR_HOME,
        AdjustedWeightedSum.__name__,
        fold_set_no,
    )
    if os.path.exists(_BIN_MODEL_FILE_PATH):
        with open(file=_BIN_MODEL_FILE_PATH, mode="rb") as fin:
            _estimator: AdjustedWeightedSum = pickle.load(fin)
            fin.close()
        # end : StreamReader()
    else:
        _estimator = AdjustedWeightedSum(model=model)
        dump_estimator(
            file_path=_BIN_MODEL_FILE_PATH,
            estimator=_estimator,
        )
    return _estimator


# end : build_estimator()


def dump_estimator(
    file_path: str,
    estimator: AdjustedWeightedSum,
) -> None:
    _BIN_MODEL_DIR_PATH = os.path.dirname(file_path)
    if not DirectoryPathValidator.exist_dir(_BIN_MODEL_DIR_PATH):
        DirectoryPathValidator.mkdir(_BIN_MODEL_DIR_PATH)
    with open(file=file_path, mode="wb") as fout:
        pickle.dump(estimator, fout)
        fout.close()
    # end : StreamWriter()


# end : dump_estimator()


def load_estimator(
    file_path: str,
) -> AdjustedWeightedSum:
    _estimator: AdjustedWeightedSum = None
    if os.path.exists(file_path):
        with open(file=file_path, mode="rb") as fin:
            _estimator: AdjustedWeightedSum = pickle.load(fin)
            fin.close()
    return _estimator


# end : load_estimator()


def load_recommender(
    file_path: str,
) -> BaseRecommender:
    _recommender: BaseRecommender = None
    if os.path.exists(file_path):
        with open(file=file_path, mode="rb") as fin:
            _recommender = pickle.load(fin)
            fin.close()
    return _recommender


# end : load_recommender()


def dump_recommender(
    file_path: str,
    recommender: BaseRecommender,
) -> None:
    _dir_path = os.path.dirname(file_path)
    if not DirectoryPathValidator.exist_dir(_dir_path):
        DirectoryPathValidator.mkdir(_dir_path)
    with open(file=file_path, mode="wb") as fout:
        pickle.dump(recommender, fout)
        fout.close()
    # end : StreamWriter()


# end : dump_recommender()


# end : recommendation()


def score_recommender(
    estimator: AdjustedWeightedSum,
) -> ScoreBasedRecommender:
    return ScoreBasedRecommender(estimator)


# end : score_based_recommendation()


def ela_recommender(
    estimator: AdjustedWeightedSum,
) -> ELABasedRecommender:
    return ELABasedRecommender(estimator)


if __name__ == "__main__":
    main()
# end : main()
