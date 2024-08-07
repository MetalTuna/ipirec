## build-in
import os
import sys
import pickle

## 3rd Pty.
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# [DEF] Environment variables
__dir_name__ = os.path.basename(os.path.dirname(__file__))
WORKSPACE_HOME = os.path.dirname(__file__).replace(f"/{__dir_name__}", "")
""".../ipirec"""
RESOURCE_DIR_PATH = f"{WORKSPACE_HOME}/resources"

# [SET] Env.append($WORKSPACE_HOME)
sys.path.append(WORKSPACE_HOME)


## Custom LIB.
from core import *
from colley import *
from ipirec import *


class ObservTagsScore:

    def __init__(self) -> None:
        self._model: BaseModel = None
        self._estimator: BaseEstimator = None

    # end : init()

    def read_estimator(
        self,
        set_no: int,
    ) -> AdjustedBiasedCorrelationEstimator:
        _file_path = self.get_estimator_pickle_path(set_no)
        estimator: AdjustedBiasedCorrelationEstimator = self._read_pickle_(_file_path)
        return estimator

    # end : public AdjustedBiasedCorrelationEstimator read_estimator()

    def read_model(
        self,
        set_no: int,
    ) -> CorrelationModel:
        _file_path = self.get_model_pickle_path(set_no)
        model: CorrelationModel = self._read_pickle_(_file_path)
        return model

    # end : public CorrelationModel read_model()

    def _read_pickle_(
        self,
        pickle_file_path: str,
    ):
        inst = None
        if not os.path.exists(pickle_file_path):
            raise FileNotFoundError()
        with open(
            file=pickle_file_path,
            mode="rb",
        ) as fin:
            inst = pickle.load(fin)
            fin.close()
        # end : StreamReader()
        return inst

    # end : protected Any read_pickle()

    @staticmethod
    def labeled_tags_dict() -> dict:
        _ifc_dict = dict()
        file_path = str.format(
            "{0}/tags_dictionary/IFC_tagging.csv",
            RESOURCE_DIR_PATH,
        )
        if not os.path.exists(file_path):
            raise FileNotFoundError()

        for _, r in pd.read_csv(file_path).iterrows():
            _name = r["토큰"]
            _label = r["tag_change"]

            _ifc_dict.update({_name: _label})
        # end : for (ifc_tags)
        return _ifc_dict

    # end : public static dict labeled_tags_dict()

    @staticmethod
    def get_model_pickle_path(
        set_no: int,
    ) -> str:
        return str.format(
            "{0}/{1}/{2}.bin",
            RESOURCE_DIR_PATH,
            CorrelationModel.__name__,
            set_no,
        )

    # end : public static str get_model_pickle_path()

    @staticmethod
    def get_estimator_pickle_path(
        set_no: int,
    ) -> str:
        return str.format(
            "{0}/{1}/{2}.bin",
            RESOURCE_DIR_PATH,
            AdjustedBiasedCorrelationEstimator.__name__,
            set_no,
        )

    # end : public static str get_estimator_pickle_path()


# end : class


if __name__ == "__main__":
    _extractor = ObservTagsScore()
    KFOLD_SET_NO = 0
    _model = _extractor.read_model(
        set_no=KFOLD_SET_NO,
    )
    _estimator = _extractor.read_estimator(
        set_no=KFOLD_SET_NO,
    )
    top_n_lists = [n for n in range(3, 37, 2)]

    for decision_type in [
        DecisionType.E_LIKE,
        DecisionType.E_PURCHASE,
    ]:
        _kwd = DecisionType.to_str(decision_type)
        _testset_file_path = str.format(
            "{0}/data/colley/test_{1}_{2}_list.csv",
            WORKSPACE_HOME,
            KFOLD_SET_NO,
            _kwd,
        )
        _sr = ScoreBasedRecommender(estimator=_estimator)
        _sr.prediction()
        evaluator = IRMetricsEvaluator(
            recommender=_sr,
            file_path=_testset_file_path,
        )
        evaluator.top_n_eval(top_n_lists)
        df: DataFrame = evaluator.evlautions_summary_df()
        print(f"[ScoreBased] {_kwd}")
        print(df)

        _ela = ELABasedRecommender(estimator=_estimator)
        _ela.prediction()
        evaluator = IRMetricsEvaluator(
            recommender=_ela,
            file_path=_testset_file_path,
        )
        evaluator.top_n_eval(top_n_lists)
        df: DataFrame = evaluator.evlautions_summary_df()
        print(f"[ELABased] {_kwd}")
        print(df)
    # end : for (dtypes)

# end : main()
