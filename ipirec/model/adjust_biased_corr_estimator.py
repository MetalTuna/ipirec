"""
[수정일] 
- 24.06.20 16:25. 실험모듈과 통합합됐고, pinpoint feedback으로 모델훈련이 변경됨.

[작성일]
- 24.05.22. AdjustedBiasedTagsCorrelationEstimator
"""

import numpy as np

from core import *
from .biased_corr_estimator import BiasedCorrelationEstimator
from .adjust_corr_estimator import AdjustCorrelationEstimator


class AdjustedBiasedCorrelationEstimator(
    BiasedCorrelationEstimator,
    AdjustCorrelationEstimator,
):
    def __init__(
        self,
        model: BaseModel,
        model_params: dict,
    ) -> None:
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

    def _estimate_(self, inst: BaseAction) -> BaseAction:
        return AdjustCorrelationEstimator._estimate_(self, inst)

    # """
    def _personalization_(
        self,
        target_decision: DecisionType,
    ) -> float:
        """
        - 요약:
            - 이 함수는 개인화 학습기능을 담당하며, fit()를 호출해 실행됩니다.
        """
        self.estimation_error_denom = 0.0
        self.estimation_error_numer = 0.0
        self.__fit__(
            target_decision=target_decision,
            n=self.frob_norm,
        )
        return self._current_mean_of_error_scores

    # end : protected float personalization()

    def _adjust_(
        self,
        inst: BaseAction,
    ) -> float:
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

        # adjust weight
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

        # adjust score
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
