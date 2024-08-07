"""
[작성일] 24.06.08. 16:31., 
- Comments.
    - Baseline보다 iter 높아야함;
- Train()
    - Pinpoint estimation; explicit feedback.
"""

import numpy as np

from core import *
from core import BaseModel
from .adjust_biased_corr_estimator import AdjustedBiasedCorrelationEstimator


class AdjustedBiasedCorrelationEstimatorRev(
    AdjustedBiasedCorrelationEstimator,
):
    """
    - Summary
        - IPIRec (Baseline) -- 5/30 Ver.
    - Methods
        - Train(): Pinpoint estimation and explicit feedback
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

    # end : init()

    def _adjust_(self, inst: BaseAction) -> float:
        user: UserEntity = self.user_dict[inst.user_id]
        item: ItemEntity = self.item_dict[inst.item_id]
        uidx = self.user_id_to_idx[inst.user_id]

        _U: np.ndarray = self.user_id_to_tags_map_dict[inst.user_id]
        _I: np.ndarray = self.item_id_to_tags_map_dict[inst.item_id]
        _F = np.matmul(_U, _I)
        # feature_map << top_n_tags_map (|T| * |T|)

        _is_fit = False

        # adjust weight
        for x_name in user.top_n_decision_tags_set:
            x_idx: int = self.tag_name_to_idx[x_name]
            for y_name in item.tags_set:
                y_idx: int = self.tag_name_to_idx[y_name]
                if x_idx == y_idx:
                    continue
                # estimation
                inst = self._estimate_(inst)
                _is_fit = inst.estimated_score == 1
                if _is_fit:
                    return 0.0
                _adjust = 1.0 if inst.estimated_score == 0.0 else 0.0

                # feedback
                _FW = self.arr_user_idx_to_weights[uidx] * _F
                _w_denom = np.sum(np.fabs(_FW))
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
        for x_name in user.top_n_decision_tags_set:
            x_idx: int = self.tag_name_to_idx[x_name]
            for y_name in item.tags_set:
                y_idx: int = self.tag_name_to_idx[y_name]
                if x_idx == y_idx:
                    continue
                # estimation
                inst = self._estimate_(inst)
                _is_fit = inst.estimated_score == 1
                if _is_fit:
                    return 0.0
                _adjust = 1.0 if inst.estimated_score == 0.0 else 0.0

                # feedback
                _FS = self.arr_tags_score * _F
                _s_denom = np.sum(np.fabs(_FS))
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


# end : class
