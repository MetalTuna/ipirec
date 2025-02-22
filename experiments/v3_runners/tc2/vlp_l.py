"""
[작성일] 24.06.24 12:12
IPIRecRev3

[수정일]
- YY.MM.DD HH:MM. Summary, comments.
"""

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
WORKSPACE_HOME = os.path.dirname(__file__).replace(f"/{__dir_name__}", "")
""".../ipirec"""
DATA_SET_HOME = f"{WORKSPACE_HOME}/data/colley"
""">>> `${WORKSPACE_HOME}`/data"""
MODEL_NAME = "IPIRecRev3"

# [IO] Binary instances paths variables
SUB_STORAGE_HOME = "/data/tghwang"
LOCAL_STORAGE_HOME = WORKSPACE_HOME
# BIN_INSTANCES_HOME = LOCAL_STORAGE_HOME
BIN_INSTANCES_HOME = SUB_STORAGE_HOME
RESOURCES_DIR_HOME = f"{BIN_INSTANCES_HOME}/resources"
""">>> `${BIN_INSTANCES_HOME}`/resources"""

# [SET] Env.append($WORKSPACE_HOME)
sys.path.append(WORKSPACE_HOME)

## custom modules
from core import *
from colley import *
from ipirec import *
from movielens import *
from rec_tags_freq import CosineItemsTagsFreqAddPenalty

DATE_STR = DirectoryPathValidator.current_datetime_str().split("_")[0].strip()
"""YYYYMMDD"""
TIME_STR = DirectoryPathValidator.current_datetime_str().split("_")[1].strip()
"""HHMMSS"""
RESULTS_SUMMARY_HOME = f"{WORKSPACE_HOME}/results/Rev3/{DATE_STR}"
""">>> `${WORKSPACE_HOME}`/results/Rev3/${YYYYMMDD}"""


def build_dataset(
    fold_set_no: int,
) -> ColleyFilteredDataSet:
    _dataset: ColleyFilteredDataSet = None
    _BIN_DATASET_FILE_PATH = str.format(
        "{0}/{1}/{2}.bin",
        RESOURCES_DIR_HOME,
        ColleyFilteredDataSet.__name__,
        fold_set_no,
    )

    if os.path.exists(_BIN_DATASET_FILE_PATH):
        with open(file=_BIN_DATASET_FILE_PATH, mode="rb") as fin:
            _dataset: ColleyFilteredDataSet = pickle.load(fin)
            fin.close()
        # end : StreamReader()
    else:
        _dataset = ColleyFilteredDataSet(
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
    dataset: ColleyDataSet,
    fold_set_no: int,
    top_n_tags: int = 5,
    co_occur_items: int = 4,
) -> IPIRecModelRev:
    _model: IPIRecModelRev = None
    if dataset == None:
        dataset = build_dataset(
            fold_set_no=fold_set_no,
        )
    _BIN_MODEL_FILE_PATH = str.format(
        "{0}/{1}/{2}.bin",
        RESOURCES_DIR_HOME,
        IPIRecModelRev.__name__,
        fold_set_no,
    )
    if os.path.exists(_BIN_MODEL_FILE_PATH):
        with open(file=_BIN_MODEL_FILE_PATH, mode="rb") as fin:
            _model: IPIRecModelRev = pickle.load(fin)
            fin.close()
        # end : StreamReader()
    else:
        _model_params = IPIRecModelRev.create_models_parameters(
            top_n_tags=top_n_tags,
            co_occur_items_threshold=co_occur_items,
        )
        _model = IPIRecModelRev(
            dataset=dataset,
            model_params=_model_params,
        )
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
    model: IPIRecModelRev,
    fold_set_no: int,
    score_learning_rate: float = 10**-2,
    score_generalization: float = 10**-4,
    weight_learning_rate: float = 10**-3,
    weight_generalization: float = 1.0,
    frob_norm: int = 1,
) -> IPIRecApproxEstimator:
    _estimator: IPIRecApproxEstimator = None
    _BIN_MODEL_FILE_PATH = str.format(
        "{0}/{1}/{2}.bin",
        RESOURCES_DIR_HOME,
        IPIRecApproxEstimator.__name__,
        fold_set_no,
    )
    if os.path.exists(_BIN_MODEL_FILE_PATH):
        with open(file=_BIN_MODEL_FILE_PATH, mode="rb") as fin:
            _estimator: IPIRecApproxEstimator = pickle.load(fin)
            fin.close()
        # end : StreamReader()
    else:
        _model_params = IPIRecApproxEstimator.create_models_parameters(
            score_learning_rate=score_learning_rate,
            score_generalization=score_generalization,
            weight_learning_rate=weight_learning_rate,
            weight_generalization=weight_generalization,
            frob_norm=frob_norm,
        )
        _estimator = IPIRecApproxEstimator(
            model=model,
            model_params=_model_params,
        )
        dump_estimator(
            file_path=_BIN_MODEL_FILE_PATH,
            estimator=_estimator,
        )
    return _estimator


# end : build_estimator()


def dump_estimator(
    file_path: str,
    estimator: IPIRecApproxEstimator,
) -> None:
    _BIN_MODEL_DIR_PATH = os.path.dirname(file_path)
    if not DirectoryPathValidator.exist_dir(_BIN_MODEL_DIR_PATH):
        DirectoryPathValidator.mkdir(_BIN_MODEL_DIR_PATH)
    with open(file=file_path, mode="wb") as fout:
        pickle.dump(estimator, fout)
        fout.close()
    # end : StreamWriter()


# end : dump_estimator()


def load_estimator(file_path: str) -> IPIRecApproxEstimator:
    _estimator: IPIRecApproxEstimator = None
    if os.path.exists(file_path):
        with open(file=file_path, mode="rb") as fin:
            _estimator: IPIRecApproxEstimator = pickle.load(fin)
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
    _post_dtype = DecisionType.E_LIKE
    _estimator: IPIRecApproxEstimator = load_estimator(binary_estimator_file_path)
    _fit_estimator: IPIRecApproxEstimator = None
    _loss_list = list()
    _loss_list.append(sys.float_info.max)
    while True:
        _L = _estimator._adjust_tags_corr_(_post_dtype)
        if min(_loss_list) < _L:
            # del _estimator
            if _fit_estimator != None:
                _estimator: IPIRecApproxEstimator = _fit_estimator
            break
        if min(_loss_list) > _L:
            # avoid Mem.leak.
            # if _fit_estimator != None:
            # del _fit_estimator
            # gc.collect()
            _fit_estimator: IPIRecApproxEstimator = copy.deepcopy(_estimator)
        _loss_list.append(_L)
    # end : while (S_reg)

    file_path = str.format(
        "{0}/{1}/{2}_{3}_{4}.bin",
        RESOURCES_DIR_HOME,
        IPIRecApproxEstimator.__name__,
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
    ## Model Args.
    _SET_NO = 0
    if len(sys.argv) == 2:
        _SET_NO = int(sys.argv[1])
    _TOP_N_TAGS = 20
    _CO_ITEMS = 4
    _LAMBDA_S = 10**-2
    _GAMMA_S = 10**-4
    _LAMBDA_W = 10**-3
    _GAMMA_W = 1.0
    _FROB_NORM = 1
    # _DV = 0.0
    # _TOP_N_ITEMS_LIST = [n for n in range(3, 37, 2)]

    _dataset = build_dataset(fold_set_no=_SET_NO)
    _model = build_model(
        dataset=_dataset,
        fold_set_no=_SET_NO,
        top_n_tags=_TOP_N_TAGS,
        co_occur_items=_CO_ITEMS,
    )
    _estimator = build_estimator(
        model=_model,
        fold_set_no=_SET_NO,
        score_learning_rate=_LAMBDA_S,
        score_generalization=_GAMMA_S,
        weight_learning_rate=_LAMBDA_W,
        weight_generalization=_GAMMA_W,
        frob_norm=_FROB_NORM,
    )

    # Est. Train.
    ## ForAll Decision types.
    _HL = list()
    _local_LS = list()
    _local_LW = list()
    __TRAINED_EST_FILE_PATH = str.format(
        "{0}/temp/trained_estimator_{1}.bin",
        BIN_INSTANCES_HOME,
        _SET_NO,
    )

    _trained_decisions_seq_list = list()
    for decision_type in DecisionType:
        ## [IMPORT] trained instances
        _trained_decisions_seq_list.append(decision_type)
        _trained_decisions_seq_str = DecisionType.list_to_kwd_str(
            _trained_decisions_seq_list
        )
        __ESTIMATOR_BIN_FILE_PATH = str.format(
            "{0}/{1}/{2}_{3}.bin",
            RESOURCES_DIR_HOME,
            IPIRecApproxEstimator.__name__,
            _SET_NO,
            _trained_decisions_seq_str,
        )
        if os.path.exists(__ESTIMATOR_BIN_FILE_PATH):
            recommendation(
                __ESTIMATOR_BIN_FILE_PATH,
                _SET_NO,
                _trained_decisions_seq_str,
            )
            continue
        # end : if (exist_binary_instances)

        _HL.clear()
        _HL.append(sys.float_info.max)

        while True:
            ### S
            _local_LS.clear()
            _local_LS.append(sys.float_info.max)
            while True:
                _ES = _estimator._adjust_tags_corr_(decision_type)
                if min(_local_LS) < _ES:
                    _estimator: IPIRecApproxEstimator = load_estimator(
                        __TRAINED_EST_FILE_PATH
                    )
                    break
                if min(_local_LS) > _ES:
                    ## TRAINED (DTypeSeq.)
                    dump_estimator(__TRAINED_EST_FILE_PATH, _estimator)
                _local_LS.append(_ES)
            # end : while (min S)

            if len(_local_LW) != 0:
                _min_ES = min(_local_LS)
                _min_EW = min(_local_LW)
                _harmonic_loss = (2 * _min_ES * _min_EW) / (_min_ES + _min_EW)
                if min(_HL) < _harmonic_loss:
                    _estimator: IPIRecApproxEstimator = load_estimator(
                        __TRAINED_EST_FILE_PATH
                    )
                    break
                if min(_HL) > _harmonic_loss:
                    dump_estimator(__TRAINED_EST_FILE_PATH, _estimator)
                _HL.append(_harmonic_loss)
            # end : if (is_repeat)

            ### S + B
            _local_LS.clear()
            _local_LS.append(sys.float_info.max)
            _estimator.__append_biases__()
            while True:
                _ES = _estimator._adjust_tags_corr_(decision_type)
                if min(_local_LS) < _ES:
                    _estimator: IPIRecApproxEstimator = load_estimator(
                        __TRAINED_EST_FILE_PATH
                    )
                    break
                if min(_local_LS) > _ES:
                    dump_estimator(__TRAINED_EST_FILE_PATH, _estimator)
                _local_LS.append(_ES)
            # end : while (min S)

            ### W
            _local_LW.clear()
            _local_LW.append(sys.float_info.max)
            while True:
                _EW = _estimator._personalization_(decision_type)
                if min(_local_LW) < _EW:
                    _estimator: IPIRecApproxEstimator = load_estimator(
                        __TRAINED_EST_FILE_PATH
                    )
                    break
                if min(_local_LW) > _EW:
                    dump_estimator(__TRAINED_EST_FILE_PATH, _estimator)
                _local_LW.append(_EW)
            # end : while (min W)
        # end : while (local_minima?)

        ## [EXPORT] trained instances
        dump_estimator(__ESTIMATOR_BIN_FILE_PATH, _estimator)
        # [EXPORT] model args.
        file_path = __ESTIMATOR_BIN_FILE_PATH.replace("bin", "ini")
        with open(file=file_path, mode="wt") as fout:
            _estimator._config_info.write(fout)
            fout.close()
        # end : StreamWriter()
    # end : for (decision_types)
# end : main()
