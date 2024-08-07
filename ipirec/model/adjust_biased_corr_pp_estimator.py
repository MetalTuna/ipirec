"""
작성일: 24.05.22. 
AdjustedBiasedTagsCorrelationEstimator 
"""

from tqdm import tqdm
import numpy as np

from core import *
from .biased_corr_estimator import BiasedCorrelationEstimator
from .adjust_corr_estimator import AdjustCorrelationEstimator


class AdjustedBiasedCorrelationPinpointEstimator(
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

    def train(
        self,
        target_decision: DecisionType = DecisionType.E_VIEW,
        n: int = 1,
        emit_iter_condition: int = 10,
        emit_diff_condition: float = 20,
    ) -> None:
        self.EMIT_ITER_THRESHOLD = emit_iter_condition
        self.EMIT_ERROR_THRESHOLD = emit_diff_condition

        print(f"{type(self).__name__}.train()")
        ## adjust score
        for _ in tqdm(
            iterable=range(self.score_iterations),
            desc="Adjust tags score",
            total=self.score_iterations,
        ):
            self._adjust_tags_corr_(target_decision)
        # end : for (adjust_score_iterations)

        ## personalization
        error_rate = 0.0
        error_list = list()
        error_list.append(error_rate)
        for _ in tqdm(
            iterable=range(self.EMIT_ITER_THRESHOLD),
            desc=f"[{DecisionType.to_str(target_decision)}]",
            total=self.EMIT_ITER_THRESHOLD,
        ):
            error_rate = self.__fit__(target_decision, n)
            error_list.append(error_rate)

            _err_dist = np.fabs(error_list[_] - error_list[_ + 1])
            if _err_dist < emit_diff_condition:
                print(f"break condition: |e - e'| = {_err_dist}.")
                print(f"|e - e'| < {emit_diff_condition}.")
                break
        # end : for (personalization_iterations)

    # end : public override void train()

    def _estimate_(self, inst: BaseAction) -> BaseAction:
        return AdjustCorrelationEstimator._estimate_(self, inst)

    def _adjust_(self, inst: BaseAction) -> float:
        """personalization (instances)"""
        user: UserEntity = self.user_dict[inst.user_id]
        item: ItemEntity = self.item_dict[inst.item_id]
        uidx = self.user_id_to_idx[inst.user_id]

        _U: np.ndarray = self.user_id_to_tags_map_dict[inst.user_id]
        _I: np.ndarray = self.item_id_to_tags_map_dict[inst.item_id]
        _F = np.matmul(_U, _I)
        # feature_map << top_n_tags_map (|T| * |T|)

        # _FW = self.arr_user_idx_to_weights[uidx] * _F
        # _FS = self.arr_tags_score * _F

        # _adjust = 1.0 if inst.estimated_score == 0.0 else 0.0

        # adjust weight
        # _w_denom = np.sum(np.fabs(_FW))
        for x_name in user.top_n_decision_tags_set:
            x_idx: int = self.tag_name_to_idx[x_name]
            for y_name in item.tags_set:
                y_idx: int = self.tag_name_to_idx[y_name]
                if x_idx == y_idx:
                    continue
                # preference prediction.
                inst = self._estimate_(inst)
                _adjust = 1.0 if inst.estimated_score == 0.0 else 0.0

                _corr = self.arr_tags_score[x_idx][y_idx]
                _weight = self.arr_user_idx_to_weights[uidx][x_idx][y_idx]
                # _weight = _FW[x_idx][y_idx]
                _w_denom = np.sum(np.fabs(self.arr_user_idx_to_weights[uidx] * _F))
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
        # _s_denom = np.sum(np.fabs(_FS))
        for x_name in user.top_n_decision_tags_set:
            x_idx: int = self.tag_name_to_idx[x_name]
            for y_name in item.tags_set:
                y_idx: int = self.tag_name_to_idx[y_name]
                if x_idx == y_idx:
                    continue
                # preference prediction.
                inst = self._estimate_(inst)
                _adjust = 1.0 if inst.estimated_score == 0.0 else 0.0

                _corr = self.arr_tags_score[x_idx][y_idx]
                _weight = self.arr_user_idx_to_weights[uidx][x_idx][y_idx]
                # _weight = _FW[x_idx][y_idx]
                numer = (self.ACTUAL_SCORE + _adjust) + (self.LEARNING_RATE * _weight)
                _s_denom = np.sum(np.fabs(self.arr_tags_score * _F))
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
