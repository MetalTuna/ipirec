"""
[작성일] 24.06.21 18:16. 분석모델 학습이 재정의됨
[수정일] 
- 24.06.26 18:58. 모델학습 규칙이 재 정의됨
"""

import os
import sys
import copy
import gc
import pickle

import numpy as np

# [DEF] Environment variables
__dir_name__ = os.path.basename(os.path.dirname(__file__))
WORKSPACE_HOME = os.path.dirname(__file__).replace(f"/ipirec/{__dir_name__}", "")
""".../ipirec"""
DATA_SET_HOME = f"{WORKSPACE_HOME}/data/colley"
""">>> `${WORKSPACE_HOME}`/data"""

sys.path.append(WORKSPACE_HOME)

from ipirec.model import (
    CorrelationModel,
    AdjustedBiasedCorrelationEstimator,
    ScoreBasedRecommender,
)
from core import *
from colley import *


class IPIRecApproxEstimator(AdjustedBiasedCorrelationEstimator):
    """
    - 요약:
        - train()에서 모델훈련이 재정의됨
        - 현재는 tendencies를 가하지 않는 구조임(가하면 성능이 더 좋겠으나, 현 구성에서의 성능관찰 목적으로 배제함)

    Args:
        AdjustedBiasedCorrelationEstimator (_type_): _description_
    """

    def __init__(
        self,
        model: BaseModel,
        model_params: dict,
    ) -> None:
        super().__init__(
            model,
            model_params,
        )
        self._KFOLD_NO = int(
            self._config_info.get(
                section="DataSet",
                option="kfold_set_no",
            )
        )

    # end : init()

    def train(
        self,
        fit_dtype_seq: list = [
            DecisionType.E_VIEW,
            DecisionType.E_LIKE,
            DecisionType.E_PURCHASE,
        ],
        post_reg_dtype_seq: list = [],
        BIN_DUMP_DIR_PATH: str = None,
    ) -> None:
        _oracle_estimator: IPIRecApproxEstimator = None
        if BIN_DUMP_DIR_PATH != None:
            _proc_kwd = DecisionType.list_to_kwd_str(fit_dtype_seq)
            _post_kwd = DecisionType.list_to_kwd_str(post_reg_dtype_seq)
            _prev_file_path = str.format(
                "{0}/{1}/{2}_{3}",
                BIN_DUMP_DIR_PATH,
                IPIRecApproxEstimator.__name__,
                self._KFOLD_NO,
                _proc_kwd,
            )
            if _post_kwd != "":
                _prev_file_path += f"_{_post_kwd}"
            _prev_file_path += ".bin"

            if os.path.exists(_prev_file_path):
                with open(_prev_file_path, "rb") as fin:
                    self: IPIRecApproxEstimator = pickle.load(fin)
                    fin.close()
                # end : StreamReader()
                return
            if not DirectoryPathValidator.exist_dir(BIN_DUMP_DIR_PATH):
                DirectoryPathValidator.mkdir(BIN_DUMP_DIR_PATH)
        # end : if BIN_DUMP_DIR_PATH != None

        # harmonic loss
        _HL = list()

        # 전처리
        ## 태그점수 보정 (DM -> ML)
        _ls = self.__fit_scores__(DecisionType.E_VIEW)
        ## 태그 축 계산 (DENSE)
        self.__append_biases__()
        ## 태그점수 보정 (GEN)
        _ls = self.__fit_scores__(DecisionType.E_VIEW)

        if BIN_DUMP_DIR_PATH != None:
            file_path = str.format(
                "{0}/{1}/{2}.bin",
                BIN_DUMP_DIR_PATH,
                IPIRecApproxEstimator.__name__,
                self._KFOLD_NO,
            )
            with open(file_path, "wb") as fout:
                pickle.dump(self, fout)
                fout.close()
            # end : StreamWriter()

        # 주처리
        _proc_dtype_seq_list = list()
        _post_dtype_seq_list = list()
        ## 의사결정 타입별 순차 보정
        for decision_type in fit_dtype_seq:
            _proc_dtype_seq_list.append(decision_type)
            _proc_kwd = DecisionType.list_to_kwd_str(_proc_dtype_seq_list)
            self._config_info.set(
                "Estimator",
                "train_seq",
                _proc_kwd,
            )

            ## LOAD_BIN
            if BIN_DUMP_DIR_PATH != None:
                file_path = str.format(
                    "{0}/{1}/{2}_{3}.bin",
                    BIN_DUMP_DIR_PATH,
                    IPIRecApproxEstimator.__name__,
                    self._KFOLD_NO,
                    _proc_kwd,
                )
                if os.path.exists(file_path):
                    with open(file_path, "rb") as fin:
                        self: IPIRecApproxEstimator = pickle.load(fin)
                        fin.close()
                    # end : StreamReader()
                    continue
            # end : if

            _min_hl = sys.float_info.max
            _HL.append(_min_hl)
            while True:
                ### 개인화 보정
                _ls = self.__fit_weights__(decision_type)
                ### 태그점수 보정
                _lw = self.__fit_scores__(decision_type)
                _hl = (2 * _ls * _lw) / (_ls + _lw)
                _min_hl = min(_HL)
                if _min_hl > _hl:
                    if _oracle_estimator != None:
                        self: IPIRecApproxEstimator = _oracle_estimator
                    break
                if _min_hl < _hl:
                    _oracle_estimator: IPIRecApproxEstimator = copy.deepcopy(self)
                _HL.append(_hl)
            # end : while (!is_fit)

            ## DUMP_BIN
            if BIN_DUMP_DIR_PATH != None:
                file_path = str.format(
                    "{0}/{1}/{2}_{3}.bin",
                    BIN_DUMP_DIR_PATH,
                    IPIRecApproxEstimator.__name__,
                    self._KFOLD_NO,
                    _proc_kwd,
                )
                with open(file_path, "wb") as fout:
                    pickle.dump(self, fout)
                    fout.close()
                # end : StreamWriter()
            # end : if
        # end : for (decision_types)
        _HL.clear()
        gc.collect()

        # 후처리
        for decision_type in post_reg_dtype_seq:
            _post_dtype_seq_list.append(decision_type)
            _post_kwd = DecisionType.list_to_kwd_str(_post_dtype_seq_list)
            ## LOAD_BIN
            if BIN_DUMP_DIR_PATH != None:
                file_path = str.format(
                    "{0}/{1}/{2}_{3}_{4}.bin",
                    BIN_DUMP_DIR_PATH,
                    IPIRecApproxEstimator.__name__,
                    self._KFOLD_NO,
                    _proc_kwd,
                    _post_kwd,
                )
                if os.path.exists(file_path):
                    with open(file_path, "rb") as fin:
                        self: IPIRecApproxEstimator = pickle.load(fin)
                        fin.close()
                    # end : StreamReader()
                    continue
            # end : if

            if decision_type != DecisionType.E_VIEW:
                self.__fit_scores__(DecisionType.E_VIEW)
            _ls = self.__fit_scores__(decision_type)

            ## DUMP_BIN
            self._config_info.set(
                "Estimator",
                "post_train_seq",
                _post_kwd,
            )
            if BIN_DUMP_DIR_PATH != None:
                file_path = str.format(
                    "{0}/{1}/{2}_{3}_{4}.bin",
                    BIN_DUMP_DIR_PATH,
                    IPIRecApproxEstimator.__name__,
                    self._KFOLD_NO,
                    _proc_kwd,
                    _post_kwd,
                )
                with open(file_path, "wb") as fout:
                    pickle.dump(self, fout)
                    fout.close()
                # end : StreamWriter()
            # end : if
        # end : for (post_decision_types)

    # public override void train()

    def __fit_scores__(
        self,
        decision_type: DecisionType = DecisionType.E_VIEW,
    ) -> float:
        _oracle_estimator: IPIRecApproxEstimator = None
        _L = list()
        _L.append(sys.float_info.max)
        while True:
            _ls = self._adjust_tags_corr_(decision_type)
            _min_loss = min(_L)
            if _min_loss < _ls:
                if _oracle_estimator != None:
                    self: IPIRecApproxEstimator = _oracle_estimator
                break
            if _min_loss > _ls:
                # cp >> mem
                _oracle_estimator: IPIRecApproxEstimator = copy.deepcopy(self)
            _L.append(_ls)
        # end : while
        _L.clear()
        gc.collect()

        return _min_loss

    # end : private float fit_scores()

    def __fit_weights__(
        self,
        decision_type: DecisionType = DecisionType.E_VIEW,
    ):
        _min_loss = sys.float_info.max
        _oracle_estimator: IPIRecApproxEstimator = None
        _L = list()
        _L.append(_min_loss)
        while True:
            _lw = self._personalization_(decision_type)
            _min_loss = min(_L)
            if _min_loss < _lw:
                if _oracle_estimator != None:
                    self: IPIRecApproxEstimator = _oracle_estimator
                break
            if _min_loss > _lw:
                _oracle_estimator: IPIRecApproxEstimator = copy.deepcopy(self)
            _L.append(_lw)
        # end : while (fit)
        _L.clear()
        gc.collect()

        if decision_type != DecisionType.E_VIEW:
            self.__fit_scores__(DecisionType.E_VIEW)
            _min_loss = self.__fit_scores__(decision_type)

        return _min_loss

    # end : private float fit_weights()

    def _set_model_params_(self, model_params: dict) -> None:
        self.score_learning_rate = model_params["score_learning_rate"]
        self.score_generalization = model_params["score_generalization"]
        self.LEARNING_RATE = model_params["weight_learning_rate"]
        self.REGULARIZATION = model_params["weight_generalization"]
        self.frob_norm = model_params["frob_norm"]
        self.default_voting = model_params["default_voting"]

    # end : protected override void set_model_params()

    @staticmethod
    def create_models_parameters(
        score_learning_rate: float,
        score_generalization: float,
        weight_learning_rate: float,
        weight_generalization: float,
        frob_norm: int = 1,
    ) -> dict:
        return AdjustedBiasedCorrelationEstimator.create_models_parameters(
            0,
            score_learning_rate,
            score_generalization,
            0,
            weight_learning_rate,
            weight_generalization,
            frob_norm,
            0.0,
        )

    # public static override dict create_models_parameters()


# end : class
