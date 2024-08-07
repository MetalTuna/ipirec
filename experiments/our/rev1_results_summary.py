## build-in
import os
import sys
import pickle

__dir_name__ = os.path.basename(os.path.dirname(__file__))
WORKSPACE_HOME = os.path.dirname(__file__).replace(f"/experiments/{__dir_name__}", "")
""".../ipirec"""
DATA_SET_HOME = f"{WORKSPACE_HOME}/data/colley"
""">>> `${WORKSPACE_HOME}`/data"""
MODEL_NAME = "IPIRec_Rev1"

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

DATE_STR = DirectoryPathValidator.current_datetime_str().split("_")[0].strip()
"""YYYYMMDD"""
TIME_STR = DirectoryPathValidator.current_datetime_str().split("_")[1].strip()
"""HHMMSS"""
RESULTS_SUMMARY_HOME = f"{WORKSPACE_HOME}/results/{DATE_STR}/{MODEL_NAME}"
""">>> `${WORKSPACE_HOME}`/results/${YYYYMMDD}/${MODEL_NAME}"""


def main():
    KFOLD_CONDITION = 5
    _BIN_REC_DIR_PATH = str.format(
        "{0}/{1}",
        RESOURCES_DIR_HOME,
        ScoreBasedRecommender.__name__,
    )
    if not os.path.exists(_BIN_REC_DIR_PATH):
        print(_BIN_REC_DIR_PATH)
        raise FileNotFoundError()

    ## TRAIN_CONDITIONS_DEF
    top_n_items = [n for n in range(3, 37, 2)]

    if not DirectoryPathValidator.exist_dir(RESULTS_SUMMARY_HOME):
        DirectoryPathValidator.mkdir(RESULTS_SUMMARY_HOME)

    _IR_RES_AGGR_FILE_PATH = str.format(
        "{0}/IRMetrics_{1}_{2}.csv",
        RESULTS_SUMMARY_HOME,
        MODEL_NAME,
        TIME_STR,
    )
    if not os.path.exists(_IR_RES_AGGR_FILE_PATH):
        with open(_IR_RES_AGGR_FILE_PATH, "wt") as fout:
            fout.write("set_no,top-n,decision_type,f1-score,hits\n")
            fout.close()
    _TS_RES_AGGR_FILE_PATH = str.format(
        "{0}/TagsScores_{1}_{2}.csv",
        RESULTS_SUMMARY_HOME,
        MODEL_NAME,
        TIME_STR,
    )
    if not os.path.exists(_TS_RES_AGGR_FILE_PATH):
        with open(_TS_RES_AGGR_FILE_PATH, "wt") as fout:
            fout.write("set_no,observ,decision_type,rmse,hits\n")
            fout.close()
    _TD_RES_AGGR_FILE_PATH = str.format(
        "{0}/TagsFreqDists_{1}_{2}.csv",
        RESULTS_SUMMARY_HOME,
        MODEL_NAME,
        TIME_STR,
    )
    if not os.path.exists(_TD_RES_AGGR_FILE_PATH):
        with open(_TD_RES_AGGR_FILE_PATH, "wt") as fout:
            fout.write("set_no,decision_type,distance\n")
            fout.close()

    with open(
        file=_IR_RES_AGGR_FILE_PATH,
        mode="at",
    ) as fout_ir, open(
        file=_TS_RES_AGGR_FILE_PATH,
        mode="at",
    ) as fout_ts, open(
        file=_TD_RES_AGGR_FILE_PATH,
        mode="at",
    ) as fout_td:
        for _KFOLD_NO in range(KFOLD_CONDITION):
            file_path = str.format(
                "{0}/{1}.bin",
                _BIN_REC_DIR_PATH,
                _KFOLD_NO,
            )
            if not os.path.exists(file_path):
                continue
            print(file_path)

            # [IMPORT] recommender
            recommender: ScoreBasedRecommender = load_recommender(
                file_path=file_path,
            )
            if recommender == None:
                exit(0)

            # Evaluation
            for decision_type in [
                DecisionType.E_LIKE,
                DecisionType.E_PURCHASE,
            ]:
                _DTYPE_STR = DecisionType.to_str(decision_type)
                file_path = str.format(
                    "{0}/test_{1}_{2}_list.csv",
                    DATA_SET_HOME,
                    _KFOLD_NO,
                    _DTYPE_STR,
                )

                ir_eval = IRMetricsEvaluator(
                    recommender=recommender,
                    file_path=file_path,
                )
                ir_eval.top_n_eval(top_n_items)
                for _, r in ir_eval.evlautions_summary_df().iterrows():
                    _TOP_N = int(r["Conditions"])
                    _F1_SCORE = float(r["F1-score"])
                    _HITS = int(r["Hits"])
                    fout_ir.write(
                        str.format(
                            "{0},{1},{2},{3},{4}\n",
                            _KFOLD_NO,
                            _TOP_N,
                            _DTYPE_STR,
                            _F1_SCORE,
                            _HITS,
                        )
                    )
                # end : for (top_n_results)

                ## RMSE
                score_eval = TagsScoreRMSEEvaluator(recommender, file_path)
                score_eval.eval()
                fout_ts.write(
                    str.format(
                        "{0},{1},{2},{3},{4}\n",
                        _KFOLD_NO,
                        "hits",
                        _DTYPE_STR,
                        str(score_eval.rmse_hits),
                        str(score_eval.no_of_hits),
                    )
                )
                fout_ts.write(
                    str.format(
                        "{0},{1},{2},{3},0\n",
                        _KFOLD_NO,
                        "all",
                        _DTYPE_STR,
                        score_eval.rmse_forall,
                    )
                )

                ## Tags Freq. Dist.
                tags_freq_eval = CosineItemsTagsFreqAddPenalty()
                tags_dist: float = tags_freq_eval.tags_freq_distance(
                    file_path, recommender
                )
                fout_td.write(
                    str.format(
                        "{0},{1},{2}\n",
                        _KFOLD_NO,
                        _DTYPE_STR,
                        tags_dist,
                    )
                )
        # end : for (kfolds)

        fout_td.close()
        fout_ts.close()
        fout_ir.close()
    # end : StreamWriter(IR, TS, TD)


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
    top_n_tags: int = 5,
    co_occur_items: int = 4,
) -> IPIRecModelSeries1:
    _model: IPIRecModelSeries1 = None
    if dataset == None:
        dataset = build_dataset(
            fold_set_no=fold_set_no,
        )
    _BIN_MODEL_FILE_PATH = str.format(
        "{0}/{1}/{2}.bin",
        RESOURCES_DIR_HOME,
        IPIRecModelSeries1.__name__,
        fold_set_no,
    )
    if os.path.exists(_BIN_MODEL_FILE_PATH):
        with open(file=_BIN_MODEL_FILE_PATH, mode="rb") as fin:
            _model: IPIRecModelSeries1 = pickle.load(fin)
            fin.close()
        # end : StreamReader()
    else:
        _model_params = IPIRecModelSeries1.create_models_parameters(
            top_n_tags=top_n_tags,
            co_occur_items_threshold=co_occur_items,
        )
        _model = IPIRecModelSeries1(
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
    model: IPIRecModelSeries1,
    fold_set_no: int,
    score_learning_rate: float = 10**-2,
    score_generalization: float = 10**-4,
    weight_learning_rate: float = 10**-3,
    weight_generalization: float = 1.0,
    frob_norm: int = 1,
) -> IPIRecEstimatorSeries1:
    _estimator: IPIRecEstimatorSeries1 = None
    _BIN_MODEL_FILE_PATH = str.format(
        "{0}/{1}/{2}.bin",
        RESOURCES_DIR_HOME,
        IPIRecEstimatorSeries1.__name__,
        fold_set_no,
    )
    if os.path.exists(_BIN_MODEL_FILE_PATH):
        with open(file=_BIN_MODEL_FILE_PATH, mode="rb") as fin:
            _estimator: IPIRecEstimatorSeries1 = pickle.load(fin)
            fin.close()
        # end : StreamReader()
    else:
        _model_params = IPIRecEstimatorSeries1.create_models_parameters(
            score_learning_rate=score_learning_rate,
            score_generalization=score_generalization,
            weight_learning_rate=weight_learning_rate,
            weight_generalization=weight_generalization,
            frob_norm=frob_norm,
        )
        _estimator = IPIRecEstimatorSeries1(
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
    estimator: IPIRecEstimatorSeries1,
) -> None:
    _BIN_MODEL_DIR_PATH = os.path.dirname(file_path)
    if not DirectoryPathValidator.exist_dir(_BIN_MODEL_DIR_PATH):
        DirectoryPathValidator.mkdir(_BIN_MODEL_DIR_PATH)
    with open(file=file_path, mode="wb") as fout:
        pickle.dump(estimator, fout)
        fout.close()
    # end : StreamWriter()


# end : dump_estimator()


def load_estimator(file_path: str) -> IPIRecEstimatorSeries1:
    _estimator: IPIRecEstimatorSeries1 = None
    if os.path.exists(file_path):
        with open(file=file_path, mode="rb") as fin:
            _estimator: IPIRecEstimatorSeries1 = pickle.load(fin)
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

if __name__ == "__main__":
    main()
# end : main()
