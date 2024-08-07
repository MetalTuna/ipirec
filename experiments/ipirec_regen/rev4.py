## build-in
import os
import sys
import pickle
import copy

# import gc

## 3rd Pty. LIB.
from pandas import DataFrame

# [DEF] Environment variables
__dir_name__ = os.path.basename(os.path.dirname(__file__))
WORKSPACE_HOME = os.path.dirname(__file__).replace(f"/experiments/{__dir_name__}", "")
DATA_SET_HOME = f"{WORKSPACE_HOME}/data/colley"
MODEL_NAME = "IPIRec_Rev4"

# [IO] Binary instances paths variables
SUB_STORAGE_HOME = "/data/tghwang"
LOCAL_STORAGE_HOME = WORKSPACE_HOME
# BIN_INSTANCES_HOME = LOCAL_STORAGE_HOME
BIN_INSTANCES_HOME = SUB_STORAGE_HOME
RESOURCES_DIR_HOME = f"{BIN_INSTANCES_HOME}/resources/{MODEL_NAME}"
""">>> `${BIN_INSTANCES_HOME}`/resources/${MODEL_NAME}"""

# [SET] Env.append($WORKSPACE_HOME)
sys.path.append(WORKSPACE_HOME)

## custom modules
from core import *
from colley import *
from ipirec import *

# from ipirec import *

# from rec_tags_freq import CosineItemsTagsFreqAddPenalty

DATE_STR = DirectoryPathValidator.current_datetime_str().split("_")[0].strip()
"""YYYYMMDD"""
TIME_STR = DirectoryPathValidator.current_datetime_str().split("_")[1].strip()
"""HHMMSS"""
RESULTS_SUMMARY_HOME = f"{WORKSPACE_HOME}/results/{DATE_STR}/{MODEL_NAME}"
""">>> `${WORKSPACE_HOME}`/results/${YYYYMMDD}/${MODEL_NAME}"""


def main():
    ## Model Args.
    _SET_NO = 0
    if len(sys.argv) == 2:
        _SET_NO = int(sys.argv[1])
    _LAMBDA_S = 10**-2
    _GAMMA_S = 10**-4
    _LAMBDA_W = 10**-3
    _GAMMA_W = 10**0
    _FROB_NORM = 1
    _TOP_N_ITEMS_LIST = [n for n in range(3, 37, 2)]

    TRAIN_SEQ = [DecisionType.E_VIEW, DecisionType.E_LIKE, DecisionType.E_PURCHASE]
    POST_TRAIN_SEQ = [DecisionType.E_VIEW]

    _dataset = build_dataset(fold_set_no=_SET_NO)
    _model = build_model(
        dataset=_dataset,
        fold_set_no=_SET_NO,
    )
    _estimator = build_estimator(
        model=_model,
        fold_set_no=_SET_NO,
        score_learning_rate=_LAMBDA_S,
        score_generalization=_GAMMA_S,
        weight_learning_rate=_LAMBDA_W,
        weight_generalization=_GAMMA_W,
        frob_norm=_FROB_NORM,
        train_seq=TRAIN_SEQ,
        post_train_seq=POST_TRAIN_SEQ,
    )
    _recommender = ScoreBasedRecommender(_estimator)
    _recommender.prediction()
    file_path = str.format(
        "{0}/{1}/{2}.bin",
        RESOURCES_DIR_HOME,
        ScoreBasedRecommender.__name__,
        _SET_NO,
    )
    dump_recommender(file_path, _recommender)
    """
    file_path = str.format(
        "{0}/{1}/{2}.bin",
        RESOURCES_DIR_HOME,
        ScoreBasedRecommender.__name__,
        _SET_NO,
    )
    _recommender: ScoreBasedRecommender = None
    if os.path.exists(file_path):
        _recommender: ScoreBasedRecommender = load_recommender(file_path)
    else:
        _recommender = ScoreBasedRecommender(_estimator)
        _recommender.prediction()
        dump_recommender(file_path, _recommender)
    """
    for decision_type in [
        DecisionType.E_LIKE,
        DecisionType.E_PURCHASE,
    ]:
        file_path = _dataset.kfold_file_path(
            kfold_set_no=_SET_NO,
            decision_type=decision_type,
            is_train_set=False,
        )
        if os.path.exists(file_path):
            _evaluator = IRMetricsEvaluator(
                recommender=_recommender,
                file_path=file_path,
            )
            _evaluator.top_n_eval(_TOP_N_ITEMS_LIST)
            df = _evaluator.evlautions_summary_df()
            print(df)
    # end : for (positive_decisions)


# end : main()


def build_dataset(
    fold_set_no: int,
) -> ColleyDataSetRev:
    _dataset: ColleyDataSetRev = None
    _BIN_DATASET_FILE_PATH = str.format(
        "{0}/{1}/{2}.bin",
        RESOURCES_DIR_HOME,
        ColleyDataSetRev.__name__,
        fold_set_no,
    )

    if os.path.exists(_BIN_DATASET_FILE_PATH):
        with open(file=_BIN_DATASET_FILE_PATH, mode="rb") as fin:
            _dataset: ColleyDataSetRev = pickle.load(fin)
            fin.close()
        # end : StreamReader()
    else:
        _dataset = ColleyDataSetRev(
            dataset_dir_path=DATA_SET_HOME,
        )
        _dataset.load_kfold_train_set(
            kfold_set_no=fold_set_no,
        )
        _dataset.append_interest_tags()
        _BIN_DATASET_DIR_PATH = os.path.dirname(_BIN_DATASET_FILE_PATH)
        if not DirectoryPathValidator.exist_dir(_BIN_DATASET_DIR_PATH):
            DirectoryPathValidator.mkdir(_BIN_DATASET_DIR_PATH)
        with open(file=_BIN_DATASET_FILE_PATH, mode="wb") as fout:
            pickle.dump(_dataset, fout)
            fout.close()
        # end : StreamWriter()
    # end : if (EXIST_BIN_DATASET)
    return _dataset


# end : build_dataset()


def build_model(
    dataset: ColleyDataSetRev,
    fold_set_no: int,
) -> IPIRecModelRev3:
    _model: IPIRecModelRev3 = None
    if dataset == None:
        dataset = build_dataset(
            fold_set_no=fold_set_no,
        )
    _BIN_MODEL_FILE_PATH = str.format(
        "{0}/{1}/{2}.bin",
        RESOURCES_DIR_HOME,
        IPIRecModelRev3.__name__,
        fold_set_no,
    )
    if os.path.exists(_BIN_MODEL_FILE_PATH):
        with open(file=_BIN_MODEL_FILE_PATH, mode="rb") as fin:
            _model: IPIRecModelRev3 = pickle.load(fin)
            fin.close()
        # end : StreamReader()
    else:
        _model = IPIRecModelRev3(dataset)
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
    model: IPIRecModelRev3,
    fold_set_no: int,
    score_learning_rate: float = 10**-2,
    score_generalization: float = 10**-4,
    weight_learning_rate: float = 10**-3,
    weight_generalization: float = 1.0,
    frob_norm: int = 1,
    train_seq: list = [d for d in DecisionType],
    post_train_seq: list = list(),
) -> IPIRecEstimatorRev3:
    _estimator: IPIRecEstimatorRev3 = None
    """
    _BIN_MODEL_FILE_PATH = str.format(
        "{0}/{1}/{2}_{3}.bin",
        RESOURCES_DIR_HOME,
        IPIRecEstimatorRev3.__name__,
        fold_set_no,
        DecisionType.list_to_kwd_str(train_seq),
    )
    """
    _BIN_MODEL_FILE_PATH = str.format(
        "{0}/{1}/{2}_{3}",
        RESOURCES_DIR_HOME,
        IPIRecEstimatorRev3.__name__,
        fold_set_no,
        DecisionType.list_to_kwd_str(train_seq),
    )
    __post_train_seq_str = DecisionType.list_to_kwd_str(post_train_seq)
    _BIN_MODEL_FILE_PATH += "" if __post_train_seq_str == "" else "_"
    _BIN_MODEL_FILE_PATH += __post_train_seq_str
    _BIN_MODEL_FILE_PATH += ".bin"

    if os.path.exists(_BIN_MODEL_FILE_PATH):
        with open(_BIN_MODEL_FILE_PATH, "rb") as fin:
            _estimator: IPIRecEstimatorRev3 = pickle.load(fin)
            fin.close()
        # end : StreamReader()
        return _estimator
    # end : if

    _BIN_MODEL_FILE_PATH = str.format(
        "{0}/{1}/{2}.bin",
        RESOURCES_DIR_HOME,
        IPIRecEstimatorRev3.__name__,
        fold_set_no,
    )
    if os.path.exists(_BIN_MODEL_FILE_PATH):
        with open(file=_BIN_MODEL_FILE_PATH, mode="rb") as fin:
            _estimator: IPIRecEstimatorRev3 = pickle.load(fin)
            fin.close()
        # end : StreamReader()
    else:
        _model_params = IPIRecEstimatorRev3.create_models_parameters(
            score_learning_rate=score_learning_rate,
            score_generalization=score_generalization,
            weight_learning_rate=weight_learning_rate,
            weight_generalization=weight_generalization,
            frob_norm=frob_norm,
        )
        _estimator = IPIRecEstimatorRev3(
            model=model,
            model_params=_model_params,
        )
        file_path = str.format(
            "{0}/{1}/{2}.bin",
            RESOURCES_DIR_HOME,
            IPIRecEstimatorRev3.__name__,
            fold_set_no,
        )
        dump_estimator(file_path, _estimator)
    _estimator = train(_estimator, train_seq, post_train_seq)

    return _estimator


# end : build_estimator()


def train(
    _estimator: IPIRecEstimatorRev3,
    train_seq: list = [d for d in DecisionType],
    post_train_seq: list = list(),
) -> IPIRecEstimatorRev3:
    _train_kwd_str = DecisionType.list_to_kwd_str(train_seq)
    _post_kwd_str = DecisionType.list_to_kwd_str(post_train_seq)
    fold_set_no = int(
        _estimator._config_info.get(
            "DataSet",
            "kfold_set_no",
        )
    )
    _file_path = str.format(
        "{0}/{1}/{2}_{3}",
        RESOURCES_DIR_HOME,
        IPIRecEstimatorRev3.__name__,
        fold_set_no,
        _train_kwd_str,
    )
    _file_path += "" if _post_kwd_str == "" else f"_{_post_kwd_str}"
    _file_path += ".bin"
    """
    _file_path = str.format(
        "{0}/{1}/{2}_{3}_{4}.bin",
        RESOURCES_DIR_HOME,
        IPIRecEstimatorRev3.__name__,
        fold_set_no,
        _train_kwd_str,
        _post_kwd_str,
    )
    """
    if os.path.exists(_file_path):
        with open(_file_path, "rb") as fin:
            _estimator: IPIRecEstimatorRev3 = pickle.load(fin)
            fin.close()
        # end : StreamReader()
        return _estimator
    # end : if

    _estimator.train(train_seq, post_train_seq, RESOURCES_DIR_HOME)
    dump_estimator(_file_path, _estimator)

    return _estimator


# end : train()


def dump_estimator(
    file_path: str,
    estimator: IPIRecEstimatorRev3,
) -> None:
    _BIN_MODEL_DIR_PATH = os.path.dirname(file_path)
    if not DirectoryPathValidator.exist_dir(_BIN_MODEL_DIR_PATH):
        DirectoryPathValidator.mkdir(_BIN_MODEL_DIR_PATH)
    with open(file_path, "wb") as fout:
        pickle.dump(estimator, fout)
        fout.close()
    # end : StreamWriter()

    with open(file_path.replace("bin", "ini"), "wt") as fout:
        estimator._config_info.write(fout)
        fout.close()
    # end : StreamWriter()


# end : dump_estimator()


def load_estimator(file_path: str) -> IPIRecEstimatorRev3:
    _estimator: IPIRecEstimatorRev3 = None
    if os.path.exists(file_path):
        with open(file=file_path, mode="rb") as fin:
            _estimator: IPIRecEstimatorRev3 = pickle.load(fin)
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


def recommendation(
    binary_estimator_file_path: str,
    kfold_set_no: int,
    trained_decisions_seq_str: str,
):
    _post_dtype = DecisionType.E_VIEW
    _estimator: IPIRecEstimatorRev3 = load_estimator(binary_estimator_file_path)
    _fit_estimator: IPIRecEstimatorRev3 = None
    _loss_list = list()
    _loss_list.append(sys.float_info.max)
    while True:
        _L = _estimator._adjust_tags_corr_(_post_dtype)
        if min(_loss_list) < _L:
            # del _estimator
            if _fit_estimator != None:
                _estimator: IPIRecEstimatorRev3 = _fit_estimator
            break
        if min(_loss_list) > _L:
            # avoid Mem.leak.
            # if _fit_estimator != None:
            # del _fit_estimator
            # gc.collect()
            _fit_estimator: IPIRecEstimatorRev3 = copy.deepcopy(_estimator)
        _loss_list.append(_L)
    # end : while (S_reg)

    file_path = str.format(
        "{0}/{1}/{2}_{3}_{4}.bin",
        RESOURCES_DIR_HOME,
        IPIRecEstimatorRev3.__name__,
        kfold_set_no,
        trained_decisions_seq_str,
        DecisionType.to_str(_post_dtype)[0],
    )
    dump_estimator(
        file_path=file_path,
        estimator=_estimator,
    )

    file_path = str.format(
        "{0}/{1}/{2}_{3}_{4}.bin",
        RESOURCES_DIR_HOME,
        ScoreBasedRecommender.__name__,
        kfold_set_no,
        trained_decisions_seq_str,
        DecisionType.to_str(_post_dtype)[0],
    )
    ## [Obs.] Raw scores;
    _recommender: ScoreBasedRecommender = load_recommender(file_path=file_path)
    if _recommender == None:
        _recommender = ScoreBasedRecommender(estimator=_estimator)
        _recommender.prediction()
        dump_recommender(
            file_path=file_path,
            recommender=_recommender,
        )
    # end : if (EXIST_RECOMMENDER_BIN)


# end : recommendation()

if __name__ == "__main__":
    main()
