"""
[작성일] 24.07.09 17:29. 마지막 분석 (구현 중)
[수정일] 
- 24.07.10 10:00. 
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

from core import *
from colley import *
from .ipirec_estimator_series2 import IPIRecEstimatorSeries2


class IPIRecEstimatorSeries3(IPIRecEstimatorSeries2):
    """
    - 요약:
        - 모델훈련이 재정의됨
        - https://www.notion.so/colleykr/24-07-09-a4343f57bc5b4e72a31c9aa264cc8c45?pvs=4
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
        self._user_id_to_tags_idx: dict = None
        self._item_id_to_tags_idx: dict = None
        self.__decision_tags__()

    # end : init()

    def __decision_tags__(self) -> None:
        if isinstance(self._dataset, ColleyDataSetRev):
            self._user_id_to_tags_idx = self._dataset.user_id_to_tag_idx_dict
            self._item_id_to_tags_idx = self._dataset.item_id_to_tag_idx_dict
            return

        _user_tags_idxs = dict()
        _item_tags_idxs = dict()

        for user_id in self.user_dict.keys():
            user: UserEntity = self.user_dict[user_id]
            _idx_set = {
                self.tag_name_to_idx[_t]
                for _t in user.top_n_decision_tags_set
                if _t in self.tag_name_to_idx
            }
            for _t in user.set_of_interest_tags:
                if _t in self.tag_name_to_idx:
                    _idx_set.add(self.tag_name_to_idx[_t])
            _user_tags_idxs.update({user_id: _idx_set})
        # end : for (users)
        self._user_id_to_tags_idx = _user_tags_idxs

        for item_id in self.item_dict.keys():
            item: ItemEntity = self.item_dict[item_id]
            _idx_set = {
                self.tag_name_to_idx[_t]
                for _t in item.tags_set
                if _t in self.tag_name_to_idx
            }
            _item_tags_idxs.update({item_id: _idx_set})
        # end : for (items)
        self._item_id_to_tags_idx = _item_tags_idxs

    # end : private void decision_tags()

    def _estimate_(self, inst: BaseAction) -> BaseAction:
        uidx: int = self.user_id_to_idx[inst.user_id]
        _numer = _denom = 0.0
        for x_idx in self._user_id_to_tags_idx[inst.user_id]:
            _weighted_prob = 1.0
            _cumulated_weight = 1.0
            for y_idx in self._item_id_to_tags_idx[inst.item_id]:
                if x_idx == y_idx:
                    continue
                __weight = self.arr_user_idx_to_weights[uidx][x_idx][y_idx]
                __score = self.model.arr_tags_score[x_idx][y_idx]
                _weighted_prob *= __weight * __score
                _cumulated_weight *= np.fabs(__weight)
            # end : for (T(i))
            _numer += _weighted_prob
            _denom += _cumulated_weight
        # end : for (T(u))

        inst.estimated_score = 0.0 if _denom == 0.0 else np.tanh(_numer / _denom)
        return inst

    # end : protected override BaseAction estimate()

    def _generalization_term_(self) -> float:
        _numer = _denom = 0.0
        _T_IDXs = list(self.tag_idx_to_name.keys())
        for user_id in self.user_dict.keys():
            uidx: int = self.user_id_to_idx.get(user_id)
            _x_idxs: set = self._user_id_to_tags_idx.get(user_id)
            _x_len = len(_x_idxs)

            if _x_len == 0:
                continue
            __ws = 0.0
            for x_idx in _x_idxs:
                for y_idx in _T_IDXs:
                    __ws += (
                        self.arr_user_idx_to_weights[uidx][x_idx][y_idx]
                        * self.arr_tags_score[x_idx][y_idx]
                    ) ** 2.0
                # end : for (forall T)
            # end : for (T(u))
            _numer += (__ws / (_x_len * self.tags_count)) ** (1 / 2)
            _denom += 1.0
        # end : for (users)

        _G = (len(self.view_list) + len(self.like_list) + len(self.purchase_list)) / (
            _denom * self.items_count
        )
        __REG = 0.0 if _denom == 0.0 else _G * (_numer / _denom)

        return __REG

    # end : protected float generalization_term()

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
        _oracle_estimator: IPIRecEstimatorSeries3 = None
        if BIN_DUMP_DIR_PATH != None:
            _proc_kwd = DecisionType.list_to_kwd_str(fit_dtype_seq)
            _post_kwd = DecisionType.list_to_kwd_str(post_reg_dtype_seq)
            _prev_file_path = str.format(
                "{0}/{1}/{2}_{3}",
                BIN_DUMP_DIR_PATH,
                IPIRecEstimatorSeries3.__name__,
                self._KFOLD_NO,
                _proc_kwd,
            )
            if _post_kwd != "":
                _prev_file_path += f"_{_post_kwd}"
            _prev_file_path += ".bin"

            if os.path.exists(_prev_file_path):
                with open(_prev_file_path, "rb") as fin:
                    self: IPIRecEstimatorSeries3 = pickle.load(fin)
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
                IPIRecEstimatorSeries3.__name__,
                self._KFOLD_NO,
            )
            if not DirectoryPathValidator.exist_dir(os.path.dirname(file_path)):
                DirectoryPathValidator.mkdir(os.path.dirname(file_path))
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
                    IPIRecEstimatorSeries3.__name__,
                    self._KFOLD_NO,
                    _proc_kwd,
                )
                if os.path.exists(file_path):
                    with open(file_path, "rb") as fin:
                        self: IPIRecEstimatorSeries3 = pickle.load(fin)
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
                        self: IPIRecEstimatorSeries3 = _oracle_estimator
                    break
                if _min_hl < _hl:
                    _oracle_estimator: IPIRecEstimatorSeries3 = copy.deepcopy(self)
                _HL.append(_hl)
            # end : while (!is_fit)

            ## DUMP_BIN
            if BIN_DUMP_DIR_PATH != None:
                file_path = str.format(
                    "{0}/{1}/{2}_{3}.bin",
                    BIN_DUMP_DIR_PATH,
                    IPIRecEstimatorSeries3.__name__,
                    self._KFOLD_NO,
                    _proc_kwd,
                )
                with open(file_path, "wb") as fout:
                    pickle.dump(self, fout)
                    fout.close()
                # end : StreamWriter()

                file_path = str.format(
                    "{0}/{1}/{2}_{3}.ini",
                    BIN_DUMP_DIR_PATH,
                    IPIRecEstimatorSeries3.__name__,
                    self._KFOLD_NO,
                    _proc_kwd,
                )
                with open(file_path, "wt") as fout:
                    self._config_info.write(fout)
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
                    IPIRecEstimatorSeries3.__name__,
                    self._KFOLD_NO,
                    _proc_kwd,
                    _post_kwd,
                )
                if os.path.exists(file_path):
                    with open(file_path, "rb") as fin:
                        self: IPIRecEstimatorSeries3 = pickle.load(fin)
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
                    IPIRecEstimatorSeries3.__name__,
                    self._KFOLD_NO,
                    _proc_kwd,
                    _post_kwd,
                )
                with open(file_path, "wb") as fout:
                    pickle.dump(self, fout)
                    fout.close()
                # end : StreamWriter()

                file_path = str.format(
                    "{0}/{1}/{2}_{3}_{4}.ini",
                    BIN_DUMP_DIR_PATH,
                    IPIRecEstimatorSeries3.__name__,
                    self._KFOLD_NO,
                    _proc_kwd,
                    _post_kwd,
                )
                with open(file_path, "wt") as fout:
                    self._config_info.write(fout)
                    fout.close()
                # end : StreamWriter()
            # end : if
        # end : for (post_decision_types)

    # public override void train()

    def __fit_scores__(
        self,
        decision_type: DecisionType = DecisionType.E_VIEW,
    ) -> float:
        _oracle_estimator: IPIRecEstimatorSeries3 = None
        _L = list()
        _L.append(sys.float_info.max)
        print(f"[{type(self).__name__}] fit_scores()")
        _ = 0
        while True:
            _ls = self._adjust_tags_corr_(decision_type) + self._generalization_term_()
            if _ % 10 == 0:
                print(_ls)
            _ += 1
            _min_loss = min(_L)
            if _min_loss < _ls:
                if _oracle_estimator != None:
                    self: IPIRecEstimatorSeries3 = _oracle_estimator
                break
            if _min_loss > _ls:
                # cp >> mem
                _oracle_estimator: IPIRecEstimatorSeries3 = copy.deepcopy(self)
            _L.append(_ls)
        # end : while

        _L.clear()
        gc.collect()

        return _min_loss

    # end : private float fit_scores()

    def _adjust_tags_corr_(
        self,
        decision_type: DecisionType = DecisionType.E_VIEW,
    ) -> float:
        """adjust tags score"""
        _kwd = DecisionType.to_str(decision_type)
        _decision_list: list = self._dataset._decision_dict[_kwd]

        self.estimation_error_denom = 0.0
        self.estimation_error_numer = 0.0
        for inst in _decision_list:
            inst: BaseAction = self._estimate_(inst)
            _err = self.ACTUAL_SCORE - inst.estimated_score
            if inst.estimated_score == 1.0:
                continue
            _x_idxs: set = self._user_id_to_tags_idx[inst.user_id]
            _y_idxs: set = self._item_id_to_tags_idx[inst.item_id]

            _denom_s = 0.0
            for x_idx in _x_idxs:
                for y_idx in _y_idxs:
                    if x_idx == y_idx:
                        continue
                    _denom_s += np.fabs(self.model.arr_tags_score[x_idx][y_idx])
                # end : for (y_idx in T(i))
            # end : for (x_idx in T(u))

            for x_idx in _x_idxs:
                for y_idx in _y_idxs:
                    if x_idx == y_idx:
                        continue
                    ## feedback
                    _corr = self.arr_tags_score[x_idx][y_idx]
                    denom = _denom_s  # << denom_s

                    __x = 0.0
                    __cnt = 0.0
                    for _idx in _y_idxs:
                        if _idx == x_idx:
                            continue
                        __x += self.model.arr_tags_score[x_idx][_idx]
                        __cnt += 1.0
                    __x /= __cnt
                    __cnt = 0.0
                    __y = 0.0
                    for _idx in _x_idxs:
                        if _idx == x_idx:
                            continue
                        __y += self.model.arr_tags_score[_idx][y_idx]
                        __cnt += 1.0
                    __y /= __cnt
                    _x = 0.0 if denom == 0.0 else (_corr / denom) * _err
                    _x += (__x + __y) / 2.0
                    _x = self.score_learning_rate * (_x / 2 if _x >= 0 else _x * 2)
                    _x = _corr + _x
                    _denom_s += self.arr_tags_score[x_idx][y_idx] - _x
                    self.model.arr_tags_score[x_idx][y_idx] = _x
                # end : for (y_idx in T(i))
            # end : for (x_idx in T(u))
            self.estimation_error_denom += 1.0
            self.estimation_error_numer += (
                np.fabs(self.ACTUAL_SCORE - inst.estimated_score) ** self.frob_norm
            )
        # end : for (decisions_list)

        self._current_mean_of_error_scores = (
            0.0
            if self.estimation_error_denom == 0.0
            else (
                (self.estimation_error_numer / self.estimation_error_denom)
                ** (1.0 / self.frob_norm)
            )
        )

        return self._current_mean_of_error_scores

    # end : protected override float adjust_tags_corr()

    def _adjust_(
        self,
        inst: BaseAction,
    ) -> float:
        uidx: int = self.user_id_to_idx[inst.user_id]
        inst = self._estimate_(inst)
        _adjust = 1.0 if inst.estimated_score == 0.0 else 0.0

        _x_idxs = self._user_id_to_tags_idx[inst.user_id]
        _y_idxs = self._item_id_to_tags_idx[inst.item_id]

        def __get_w_denom__() -> float:
            __weight = 0.0
            for __x in _x_idxs:
                for __y in _y_idxs:
                    if __x == __y:
                        continue
                    __weight += np.fabs(self.arr_user_idx_to_weights[uidx][__x][__y])
            return __weight

        def __get_s_denom__() -> float:
            __score = 0.0
            for __x in _x_idxs:
                for __y in _y_idxs:
                    if __x == __y:
                        continue
                    __score += np.fabs(self.model.arr_tags_score[__x][__y])
            return __score

        # adjust weight
        _w_denom = __get_w_denom__()
        for x_idx in _x_idxs:
            for y_idx in _y_idxs:
                if x_idx == y_idx:
                    continue
                _corr = self.model.arr_tags_score[x_idx][y_idx]
                _weight = self.arr_user_idx_to_weights[uidx][x_idx][y_idx]
                numer = (self.ACTUAL_SCORE + _adjust) + (self.LEARNING_RATE * _corr)
                denom = (inst.estimated_score + _adjust) + (
                    self.REGULARIZATION * _w_denom
                )
                self.arr_user_idx_to_weights[uidx][x_idx][y_idx] = _weight * (
                    numer / denom
                )
            # end : for (y_idxs in T(i))
        # end : for (x_idxs in T(u))

        # adjust score
        _s_denom = __get_s_denom__()
        for x_idx in _x_idxs:
            for y_idx in _y_idxs:
                if x_idx == y_idx:
                    continue
                _corr = self.model.arr_tags_score[x_idx][y_idx]
                _weight = self.arr_user_idx_to_weights[uidx][x_idx][y_idx]
                numer = (self.ACTUAL_SCORE + _adjust) + (self.LEARNING_RATE * _weight)
                denom = (inst.estimated_score + _adjust) + (
                    self.REGULARIZATION * _s_denom
                )
                self.model.arr_tags_score[x_idx][y_idx] = _corr * (numer / denom)
            # end : for (y_idxs in T(i))
        # end : for (x_idxs in T(u))
        return (self.ACTUAL_SCORE - inst.estimated_score) ** self.frob_norm

    # end : protected override float adjust()

    def __fit_weights__(
        self,
        decision_type: DecisionType = DecisionType.E_VIEW,
    ):
        _min_loss = sys.float_info.max
        _oracle_estimator: IPIRecEstimatorSeries3 = None
        _L = list()
        _L.append(_min_loss)
        print(f"[{type(self).__name__}] fit_weights()")
        _ = 0
        while True:
            _lw = self._personalization_(decision_type) + self._generalization_term_()
            if _ % 10 == 0:
                print(_lw)
            _ += 1
            _min_loss = min(_L)
            if _min_loss < _lw:
                if _oracle_estimator != None:
                    self: IPIRecEstimatorSeries3 = _oracle_estimator
                break
            if _min_loss > _lw:
                _oracle_estimator: IPIRecEstimatorSeries3 = copy.deepcopy(self)
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
        score_learning_rate: float = 10**-2,
        score_generalization: float = 10**-4,
        weight_learning_rate: float = 10**-3,
        weight_generalization: float = 10**0,
        frob_norm: int = 1,
    ) -> dict:
        return IPIRecEstimatorSeries2.create_models_parameters(
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
