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
RESOURCES_DIR_HOME = f"{BIN_INSTANCES_HOME}/resources/{MODEL_NAME}"
""">>> `${BIN_INSTANCES_HOME}`/resources/${MODEL_NAME}"""
DATE_STR = DirectoryPathValidator.current_datetime_str().split("_")[0].strip()
"""YYYYMMDD"""
TIME_STR = DirectoryPathValidator.current_datetime_str().split("_")[1].strip()
"""HHMMSS"""
RESULTS_SUMMARY_HOME = f"{WORKSPACE_HOME}/results/{DATE_STR}/{MODEL_NAME}"
""">>> `${WORKSPACE_HOME}`/results/${YYYYMMDD}/${MODEL_NAME}"""
### [EMIT]


def main() -> None:
    _K_FOLD = 5
    for k in range(_K_FOLD):
        dataset: ColleyDataSet = build_dataset(k)
        model: Pearson = build_model()
    # end : for (KFold)


# end : main()


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
    top_n_tags: int = 5,
    co_occur_items: int = 4,
) -> IPIRecModel:
    _model: IPIRecModel = None
    if dataset == None:
        dataset = build_dataset(
            fold_set_no=fold_set_no,
        )
    _BIN_MODEL_FILE_PATH = str.format(
        "{0}/{1}/{2}.bin",
        RESOURCES_DIR_HOME,
        IPIRecModel.__name__,
        fold_set_no,
    )
    if os.path.exists(_BIN_MODEL_FILE_PATH):
        with open(file=_BIN_MODEL_FILE_PATH, mode="rb") as fin:
            _model: IPIRecModel = pickle.load(fin)
            fin.close()
        # end : StreamReader()
    else:
        _model_params = IPIRecModel.create_models_parameters(
            top_n_tags=top_n_tags,
            co_occur_items_threshold=co_occur_items,
        )
        _model = IPIRecModel(
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
    model: IPIRecModel,
    fold_set_no: int,
    score_learning_rate: float = 10**-2,
    score_generalization: float = 10**-4,
    weight_learning_rate: float = 10**-3,
    weight_generalization: float = 1.0,
    frob_norm: int = 1,
) -> IPIRecProbabilityEstimatorRev:
    _estimator: IPIRecProbabilityEstimatorRev = None
    _BIN_MODEL_FILE_PATH = str.format(
        "{0}/{1}/{2}.bin",
        RESOURCES_DIR_HOME,
        IPIRecProbabilityEstimatorRev.__name__,
        fold_set_no,
    )
    if os.path.exists(_BIN_MODEL_FILE_PATH):
        with open(file=_BIN_MODEL_FILE_PATH, mode="rb") as fin:
            _estimator: IPIRecProbabilityEstimatorRev = pickle.load(fin)
            fin.close()
        # end : StreamReader()
    else:
        _model_params = IPIRecProbabilityEstimatorRev.create_models_parameters(
            score_learning_rate=score_learning_rate,
            score_generalization=score_generalization,
            weight_learning_rate=weight_learning_rate,
            weight_generalization=weight_generalization,
            frob_norm=frob_norm,
        )
        _estimator = IPIRecProbabilityEstimatorRev(
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
    estimator: IPIRecProbabilityEstimatorRev,
) -> None:
    _BIN_MODEL_DIR_PATH = os.path.dirname(file_path)
    if not DirectoryPathValidator.exist_dir(_BIN_MODEL_DIR_PATH):
        DirectoryPathValidator.mkdir(_BIN_MODEL_DIR_PATH)
    with open(file=file_path, mode="wb") as fout:
        pickle.dump(estimator, fout)
        fout.close()
    # end : StreamWriter()


# end : dump_estimator()


def load_estimator(file_path: str) -> IPIRecProbabilityEstimatorRev:
    _estimator: IPIRecProbabilityEstimatorRev = None
    if os.path.exists(file_path):
        with open(file=file_path, mode="rb") as fin:
            _estimator: IPIRecProbabilityEstimatorRev = pickle.load(fin)
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
    _estimator: IPIRecProbabilityEstimatorRev = load_estimator(
        binary_estimator_file_path
    )
    _fit_estimator: IPIRecProbabilityEstimatorRev = None
    _loss_list = list()
    _loss_list.append(sys.float_info.max)
    while True:
        _L = _estimator._adjust_tags_corr_(_post_dtype)
        if min(_loss_list) < _L:
            # del _estimator
            if _fit_estimator != None:
                _estimator: IPIRecProbabilityEstimatorRev = _fit_estimator
            break
        if min(_loss_list) > _L:
            # avoid Mem.leak.
            # if _fit_estimator != None:
            # del _fit_estimator
            # gc.collect()
            _fit_estimator: IPIRecProbabilityEstimatorRev = copy.deepcopy(_estimator)
        _loss_list.append(_L)
    # end : while (S_reg)

    file_path = str.format(
        "{0}/{1}/{2}_{3}_{4}.bin",
        RESOURCES_DIR_HOME,
        IPIRecProbabilityEstimatorRev.__name__,
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
    _LAMBDA_S = 10**-3
    _GAMMA_S = 10**-4
    _LAMBDA_W = 10**-3
    _GAMMA_W = 1.0
    _FROB_NORM = 1
    # _DV = 0.0
    _TOP_N_ITEMS_LIST = [n for n in range(3, 37, 2)]

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
    fit_dtype_seq = [d for d in DecisionType]
    post_reg_dtype_seq = [DecisionType.E_PURCHASE]
    _estimator.train(
        fit_dtype_seq=fit_dtype_seq,
        post_reg_dtype_seq=post_reg_dtype_seq,
        BIN_DUMP_DIR_PATH=RESOURCES_DIR_HOME,
    )

    file_path = str.format(
        "{0}/{1}/{2}_{3}_{4}.bin",
        RESOURCES_DIR_HOME,
        ScoreBasedRecommender.__name__,
        _SET_NO,
        DecisionType.list_to_kwd_str(fit_dtype_seq),
        DecisionType.list_to_kwd_str(post_reg_dtype_seq),
    )
    _recommender: ScoreBasedRecommender = None
    if os.path.exists(file_path):
        _recommender: ScoreBasedRecommender = load_recommender(file_path)
    else:
        _recommender = ScoreBasedRecommender(_estimator)
        _recommender.prediction()
        dump_recommender(file_path, _recommender)

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
# end : main()
