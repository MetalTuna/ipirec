"""
작성일: 24.05.22.
AdjustedBiasedCorrelationApproxEstimator

- 수정일: 
    - 24.06.03. 설계오류에 대한 확인을 위한 작업
        - np.ndarray 참조오류를 수정
        - 개인화 이전 단계에서 점수보정이 이뤄지니 W, S의 기하학적 관계가 사라져서 오차가 발생한 것이 의심됨
        - 그래서 태그점수 보정은 최초호출만 이뤄지도록 수정함. 
"""

import numpy as np

from core import *
from .biased_corr_estimator import BiasedCorrelationEstimator
from .adjust_corr_estimator import AdjustCorrelationEstimator


class AdjustedBiasedCorrelationEstimatorRev3(
    BiasedCorrelationEstimator,
    AdjustCorrelationEstimator,
):
    """
    - 요약:
        - W, S의 직교관계 찾기 문제로 개정된 모듈임
        - protected float adjust()는 approximated score로 feedback함.
        - adjust_tags_score()는 최초 실행시, views를 대상으로 실행함.

    - 상속한 클래스:
        - BiasedCorrelationEstimator (_type_): _description_
        - AdjustCorrelationEstimator (_type_): _description_
    """

    def __init__(
        self,
        model: BaseModel,
        model_params: dict,
    ) -> None:
        self._is_preprocessed = False
        super(BiasedCorrelationEstimator, self).__init__(
            model,
            model_params,
        )
        self.__append_biases__()
        super(AdjustCorrelationEstimator, self).__init__(
            model,
            model_params,
        )

    # end : init()

    def _preprocess_(self):
        if self._is_preprocessed:
            return
        super()._preprocess_()
        self._is_preprocessed = True

    # end : protected override Any preprocess()

    def _estimate_(self, inst: BaseAction) -> BaseAction:
        return AdjustCorrelationEstimator._estimate_(self, inst)

    # end : protected override BaseAction estimate()

    def _adjust_tags_corr_(self) -> None:
        for inst in self.view_list:
            inst: BaseAction
            user: UserEntity = self.user_dict[inst.user_id]
            item: ItemEntity = self.item_dict[inst.item_id]

            _U: np.ndarray = self.user_id_to_tags_map_dict[inst.user_id]
            _I: np.ndarray = self.item_id_to_tags_map_dict[inst.item_id]

            ## 행렬곱
            _F = np.matmul(_U, _I)
            """feature_map << top_n_tags_map (|T| * |T|)"""
            # _C = np.matmul(1 * np.logical_not(_U == 1), _I)
            # """contains_map << item_tags_map (|T| * |T|)"""

            _is_fit = False
            for x_name in user.top_n_decision_tags_set:
                if _is_fit:
                    break
                x_idx: int = self.tag_name_to_idx[x_name]
                for y_name in item.tags_set:
                    y_idx: int = self.tag_name_to_idx[y_name]
                    if x_idx == y_idx:
                        continue
                    inst = self._estimate_(inst)

                    # is_fit?
                    if inst.estimated_score == 1.0:
                        _is_fit = True
                        break

                    ## feedback
                    _err = self.ACTUAL_SCORE - inst.estimated_score
                    _corr = self.arr_tags_score[x_idx][y_idx]
                    _FS = self.arr_tags_score * _F
                    denom = np.sum(np.fabs(_FS))
                    _x = 0.0 if denom == 0.0 else (_corr / denom) * _err
                    _x = _x / 2 if _x >= 0 else _x * 2
                    _x = self.score_learning_rate * _x
                    _b = self.score_generalization * (
                        (np.sum(_FS[x_idx][:]) + np.sum(_FS[:][y_idx])) / 2
                    )
                    self.arr_tags_score[x_idx][y_idx] = _corr + _x + _b

                    # numer = (self.ACTUAL_SCORE + __adjust_score) +
                # end : for (T(i))
            # end : for (NT(u))
        # end : for (views)

    # end : protected override void adjust_tags_corr()

    # """
    def _adjust_(self, inst: BaseAction) -> float:
        user: UserEntity = self.user_dict[inst.user_id]
        item: ItemEntity = self.item_dict[inst.item_id]
        uidx: int = self.user_id_to_idx[inst.user_id]

        _U: np.ndarray = self.user_id_to_tags_map_dict[inst.user_id]
        _I: np.ndarray = self.item_id_to_tags_map_dict[inst.item_id]
        _F = np.matmul(_U, _I)
        # feature_map << top_n_tags_map (|T| * |T|)
        _FW = self.arr_user_idx_to_weights[uidx] * _F
        _FS = self.arr_tags_score * _F
        _adjust = 1.0 if inst.estimated_score == 0.0 else 0.0

        ## adjust weight
        _w_denom = np.sum(np.fabs(_FW))
        for x_name in user.top_n_decision_tags_set:
            x_idx: int = self.tag_name_to_idx[x_name]
            for y_name in item.tags_set:
                y_idx: int = self.tag_name_to_idx[y_name]
                if x_idx == y_idx:
                    continue
                _corr = self.arr_tags_score[x_idx][y_idx]
                _weight = _FW[x_idx][y_idx]
                numer = (self.ACTUAL_SCORE + _adjust) + (self.LEARNING_RATE * _corr)
                denom = (inst.estimated_score + _adjust) + (
                    self.REGULARIZATION * _w_denom
                )
                self.arr_user_idx_to_weights[uidx][x_idx][y_idx] = _weight * (
                    numer / denom
                )
            # end : for (T(i))
        # end : for (NT(u))

        ## adjust score
        # inst = self._estimate_(inst)
        _FW = self.arr_user_idx_to_weights[uidx] * _F
        _FS = self.arr_tags_score * _F
        _s_denom = np.sum(np.fabs(_FS))
        for x_name in user.top_n_decision_tags_set:
            x_idx: int = self.tag_name_to_idx[x_name]
            for y_name in item.tags_set:
                y_idx: int = self.tag_name_to_idx[y_name]
                if x_idx == y_idx:
                    continue
                _corr = self.arr_tags_score[x_idx][y_idx]
                _weight = _FW[x_idx][y_idx]
                numer = (self.ACTUAL_SCORE + _adjust) + (self.LEARNING_RATE * _weight)
                denom = (inst.estimated_score + _adjust) + (
                    self.REGULARIZATION * _s_denom
                )
                self.arr_tags_score[x_idx][y_idx] = _corr * (numer / denom)
            # end : for (T(i))
        # end : for (NT(u))

        # inst = self._estimate_(inst)
        return (self.ACTUAL_SCORE - inst.estimated_score) ** self.frob_norm

    # end : protected override float adjust()
    # """


# end : class
