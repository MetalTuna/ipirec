"""
[작성일] 24.07.09 10:20.
- NMF
"""

## build-in
import os
import sys
import pickle

# [DEF] Environment variables
__dir_name__ = os.path.basename(os.path.dirname(__file__))
WORKSPACE_HOME = os.path.dirname(__file__).replace(f"/experiments/{__dir_name__}", "")
""".../ipirec"""
DATA_SET_HOME = f"{WORKSPACE_HOME}/data/colley"
""">>> `${WORKSPACE_HOME}`/data"""
MODEL_NAME = "NMF"

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
from decompositions import *

DATE_STR = DirectoryPathValidator.current_datetime_str().split("_")[0].strip()
"""YYYYMMDD"""
TIME_STR = DirectoryPathValidator.current_datetime_str().split("_")[1].strip()
"""HHMMSS"""
RESULTS_SUMMARY_HOME = f"{WORKSPACE_HOME}/results/{DATE_STR}/{MODEL_NAME}"
""">>> `${WORKSPACE_HOME}`/results/${YYYYMMDD}/${MODEL_NAME}"""


def main():
    ### Fixed Mod. Opt. ###
    FACTORS_DIM = 200
    FACTORIZER_ITERS = 200
    LAMBDA = 10 ** (-1 * 4)
    GAMMA = 10 ** (-1 * 5)
    FEEDBACK_ITERS = 150

    ## Train. OPT.
    FROB_NORM = 1
    DTYPE_SEQ = [
        DecisionType.E_VIEW,
        DecisionType.E_LIKE,
        DecisionType.E_PURCHASE,
    ]
    ## Model Args.
    _SET_NO = 0

    if len(sys.argv) == 2:
        _SET_NO = int(sys.argv[1])
    _TOP_N_ITEMS_LIST = [n for n in range(3, 37, 2)]

    _dataset = build_dataset(fold_set_no=_SET_NO)
    _model = build_model(
        dataset=_dataset,
        fold_set_no=_SET_NO,
        factors_dim=FACTORS_DIM,
        factors_iter=FACTORIZER_ITERS,
    )
    _estimator = build_estimator(
        model=_model,
        fold_set_no=_SET_NO,
        train_iter=FEEDBACK_ITERS,
        learning_rate=LAMBDA,
        generalization=GAMMA,
        frob_norm_n=FROB_NORM,
        decision_type_seq_list=DTYPE_SEQ,
    )

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
    factors_dim: int,
    factors_iter: int,
) -> NMFDecompositionModel:
    _model: NMFDecompositionModel = None
    if dataset == None:
        dataset = build_dataset(
            fold_set_no=fold_set_no,
        )
    _BIN_MODEL_FILE_PATH = str.format(
        "{0}/{1}/{2}.bin",
        RESOURCES_DIR_HOME,
        NMFDecompositionModel.__name__,
        fold_set_no,
    )
    if os.path.exists(_BIN_MODEL_FILE_PATH):
        with open(file=_BIN_MODEL_FILE_PATH, mode="rb") as fin:
            _model: NMFDecompositionModel = pickle.load(fin)
            fin.close()
        # end : StreamReader()
    else:
        model_params = NMFDecompositionModel.create_models_parameters(
            factors_dim=factors_dim,
            factorizer_iters=factors_iter,
        )
        _model = NMFDecompositionModel(
            dataset=dataset,
            model_params=model_params,
        )
        _model.analysis()
        _BIN_MODEL_DIR_PATH = os.path.dirname(_BIN_MODEL_FILE_PATH)
        if not DirectoryPathValidator.exist_dir(_BIN_MODEL_DIR_PATH):
            DirectoryPathValidator.mkdir(_BIN_MODEL_DIR_PATH)
        with open(file=_BIN_MODEL_FILE_PATH, mode="wb") as fout:
            pickle.dump(_model, fout)
            fout.close()
        # end : StreamWriter()

        with open(_BIN_MODEL_FILE_PATH.replace("bin", "ini"), "wt") as fout:
            _model._config_info.write(fout)
            fout.close()
    # end : if (EXIST_BIN_MODEL)

    return _model


# end : build_model()


def build_estimator(
    model: NMFDecompositionModel,
    fold_set_no: int,
    train_iter: int,
    learning_rate: float,
    generalization: float,
    frob_norm_n: int,
    decision_type_seq_list: list = [
        DecisionType.E_VIEW,
        DecisionType.E_PURCHASE,
    ],
) -> DecompositionsEstimator:
    _estimator: DecompositionsEstimator = None
    _BIN_MODEL_FILE_PATH = str.format(
        "{0}/{1}/{2}.bin",
        RESOURCES_DIR_HOME,
        DecompositionsEstimator.__name__,
        fold_set_no,
    )
    estimator_params = DecompositionsEstimator.create_models_parameters(
        learning_rate=learning_rate,
        generalization=generalization,
        train_iters=train_iter,
        train_decision_types_seq=decision_type_seq_list,
        frob_norm=frob_norm_n,
    )
    _estimator = DecompositionsEstimator(
        model=model,
        model_params=estimator_params,
    )
    # end : if

    _estimator.train()

    dump_estimator(
        file_path=_BIN_MODEL_FILE_PATH,
        estimator=_estimator,
    )

    return _estimator


# end : build_estimator()


def dump_estimator(
    file_path: str,
    estimator: DecompositionsEstimator,
) -> None:
    _BIN_MODEL_DIR_PATH = os.path.dirname(file_path)
    if not DirectoryPathValidator.exist_dir(_BIN_MODEL_DIR_PATH):
        DirectoryPathValidator.mkdir(_BIN_MODEL_DIR_PATH)
    with open(file=file_path, mode="wb") as fout:
        pickle.dump(estimator, fout)
        fout.close()
    # end : StreamWriter()

    with open(file_path.replace("bin", "ini"), "wt") as fout:
        estimator._config_info.write(fout)
        fout.close()


# end : dump_estimator()


def load_estimator(file_path: str) -> DecompositionsEstimator:
    _estimator: DecompositionsEstimator = None
    if os.path.exists(file_path):
        with open(file=file_path, mode="rb") as fin:
            _estimator: DecompositionsEstimator = pickle.load(fin)
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
        _recommender.prediction()
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

if __name__ == "__main__":
    main()
