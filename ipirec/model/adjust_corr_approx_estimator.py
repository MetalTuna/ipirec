"""
[작성일] 24.05.20.
- 태그점수 보정에 변화를 줌
    - 전역: 태그점수를 직접 변경
    - 개인: 보정 값이 가해진 재예측 결과에 대한 오차정도를 다음 변수에 가하도록 수정 
[수정사항]
- 24.05.29 11:21. 태그점수보정 변경 (너무 오래걸리므로, 한번 예측한 값의 오차를 예측에 참여한 값들의 보정에 사용하도록 수정)
- 24.05.27 15:28. 모델 피드백 변경 (처음 예측된 의사결정 값을 의사결정 보정 정도로 합니다.)
    - 이전의 모듈에서는 의사결정 예측에 참여한 변수 값들의 보정치를 반영하려고, 변경된 예측정도를 다시 구했음. 
- 24.05.22 10:55. 예측이 너무 오래걸려서, 예측함수에서 numpy함수를 사용하지 않도록 변경함
- 24.05.21 16:53. reshape로 인한 bottleneck확인 및 제거를 위해 해당부분들을 수정함
- 24.05.21 15:58. 구상안대로 구현했음 
"""

import numpy as np

from core import BaseModel, BaseAction, UserEntity, ItemEntity
from .adjust_corr_estimator import AdjustCorrelationEstimator


class AdjustCorrelationApproxEstimator(AdjustCorrelationEstimator):
    """
    - 요약:
        - 처음 예측된 의사결정 값을 의사결정 보정 정도로 하는 추정기입니다.
    """

    def __init__(self, model: BaseModel, model_params: dict) -> None:
        super().__init__(model, model_params)

    # end : init()

    def _adjust_tags_corr_(self) -> None:
        for inst in self.view_list:
            inst: BaseAction
            user: UserEntity = self.user_dict[inst.user_id]
            item: ItemEntity = self.item_dict[inst.item_id]
            _U: np.ndarray = self.user_id_to_tags_map_dict[inst.user_id]
            _I: np.ndarray = self.item_id_to_tags_map_dict[inst.item_id]

            ## 행렬곱
            """
            _F = np.matmul(_U, _I)
            # feature_map << top_n_tags_map (|T| * |T
            _C = np.matmul(1 * np.logical_not(_U == 1), _I)
            # contains_map << item_tags_map (|T| * |T

            _FS = self.arr_tags_score * _F
            _DS = self.default_voting * _C
            _E = _FS + _DS
            numer = np.sum(_E)
            denom = np.sum(np.fabs(_FS))
            inst.estimated_score = 0.0 if denom == 0.0 else numer / denom
            """
            inst = self._estimate_(inst)

            # is_fit
            if inst.estimated_score == 1.0:
                continue

            # feedback
            _F = np.matmul(_U, _I)
            _FS = self.arr_tags_score * _F
            denom = np.sum(np.fabs(_FS))
            _err = self.ACTUAL_SCORE - inst.estimated_score
            for x_name in user.top_n_decision_tags_set:
                x_idx: int = self.tag_name_to_idx[x_name]
                _norm = np.sum(_F[x_idx][:])
                _BX = 0.0 if _norm == 0.0 else np.sum(_FS[x_idx][:]) / _norm
                for y_name in item.tags_set:
                    y_idx: int = self.tag_name_to_idx[y_name]
                    if x_idx == y_idx:
                        continue
                    _corr = self.arr_tags_score[x_idx][y_idx]
                    _x = 0.0 if denom == 0.0 else (_corr / denom) * _err
                    _norm = np.sum(_F[:][y_idx])
                    _BY = 0.0 if _norm == 0.0 else np.sum(_FS[:][y_idx]) / _norm
                    # _x = (self.REGULARIZATION / 2) * (_BX + _BY)
                    # _x = _x / 2 if _x >= 0 else _x * 2
                    _x = (self.REGULARIZATION / 2) * (_BX + _BY)
                    _x *= self.LEARNING_RATE
                    self.arr_tags_score[x_idx][y_idx] = _corr + _x
                    """
                    _corr = self.arr_tags_score[x_idx][y_idx]
                    _x = 0.0 if denom == 0.0 else (_corr / denom) * _err
                    _x = _x / 2 if _x >= 0 else _x * 2
                    _x = self.LEARNING_RATE * _x
                    _b = (self.REGULARIZATION / 2) * (_BX + np.sum(_FS[:][y_idx]))
                    self.arr_tags_score[x_idx][y_idx] = _corr + _x + _b
                    """

                    # numer = (self.ACTUAL_SCORE + __adjust_score) +
                # end : for (T(i))
            # end : for (NT(u))
        # end : for (views)

    # end : protected override void adjust_tags_corr()

    def _adjust_(self, inst: BaseAction) -> float:
        uidx: int = self.user_id_to_idx[inst.user_id]
        user: UserEntity = self.user_dict[inst.user_id]
        item: ItemEntity = self.item_dict[inst.item_id]
        _U: np.ndarray = self.user_id_to_tags_map_dict[inst.user_id]
        _I: np.ndarray = self.item_id_to_tags_map_dict[inst.item_id]

        inst = self._estimate_(inst)
        if inst.estimated_score == 1.0:
            return 0.0

        ## 행렬곱
        _F = np.matmul(_U, _I)
        _FW = self.arr_user_idx_to_weights[uidx] * _F
        _FS = self.arr_tags_score * _F
        """feature_map << top_n_tags_map (|T| * |T|)"""

        _adjust = 1.0 if inst.estimated_score == 0.0 else 0.0

        dist_denom = (inst.estimated_score + _adjust) + (
            self.REGULARIZATION * np.sum(np.fabs(_FW))
        )
        corr_denom = (inst.estimated_score + _adjust) + (
            self.REGULARIZATION * np.sum(np.fabs(_FS))
        )

        ### update feedback
        for x_name in user.top_n_decision_tags_set:
            x_idx: int = self.tag_name_to_idx[x_name]
            for y_name in item.tags_set:
                y_idx: int = self.tag_name_to_idx[y_name]
                if x_idx == y_idx:
                    continue
                # Adjust distance
                _weight = self.arr_user_idx_to_weights[uidx][x_idx][y_idx]
                _corr = self.arr_tags_score[x_idx][y_idx]
                numer = (self.ACTUAL_SCORE + _adjust) + (self.LEARNING_RATE * _corr)
                _weight = _weight * (numer / dist_denom)
                self.arr_user_idx_to_weights[uidx][x_idx][y_idx] = _weight

                # Adjust score
                _weight = self.arr_user_idx_to_weights[uidx][x_idx][y_idx]
                _corr = self.arr_tags_score[x_idx][y_idx]
                numer = (self.ACTUAL_SCORE + _adjust) + (self.LEARNING_RATE * _weight)
                _corr = _corr * (numer / corr_denom)
                self.arr_tags_score[x_idx][y_idx] = _corr
            # end : for (T(i))
        # end : for (NT(u))
        return (self.ACTUAL_SCORE - inst.estimated_score) ** self.frob_norm

    # end : protected override float adjust()


# end : class
