import math

import numpy as np

from core import BaseAction, BaseTrain, UserEntity, ItemEntity
from .correlation_model import CorrelationModel


class ConditionalEstimator(BaseTrain):
    """
    사용하지 않음
    =====
    """

    def __init__(
        self,
        model: CorrelationModel,
    ) -> None:
        super().__init__(model)

    # end : init()

    def _estimate_(
        self,
        inst: BaseAction,
    ) -> BaseAction:
        numer, denom = 0.0, 0.0
        user: UserEntity = self.user_dict[inst.user_id]
        item: ItemEntity = self.item_dict[inst.item_id]
        uidx = self.user_id_to_idx[user.user_id]
        for x_name in user.top_n_decision_tags_set:
            if not x_name in self.tag_name_to_idx:
                continue
            x_idx = self.tag_name_to_idx[x_name]

            for y_name in item.tags_set:
                if not y_name in self.tag_name_to_idx:
                    continue
                y_idx = self.tag_name_to_idx[y_name]
                # x -> y
                """
                corr = (
                    self.arr_tags_score[x_idx][y_idx]
                    * self.arr_user_idx_to_weights[uidx][x_idx][y_idx]
                )
                """
                corr = (
                    self.arr_tags_score[x_idx][y_idx]
                    * self.arr_user_idx_to_weights[uidx][x_idx][y_idx]
                )
                numer += corr if y_name in user.top_n_decision_tags_set else 0.0
                denom += math.fabs(corr)
            # end : for (T(i))
        # end : for (NT(u))
        inst.estimated_score = 0.0 if denom == 0.0 else numer / denom

        return inst

    # end : protected override BaseAction estimate()

    def _adjust_(
        self,
        inst: BaseAction,
    ) -> float:
        if inst.estimated_score == 1.0:
            return 0.0
        error_rate: float = 1.0 - inst.estimated_score
        _feedback = error_rate * self.LEARNING_RATE

        if _feedback != 0.0:
            print()

        # 예측에 사용된 변수들을 순회하며 오차에 따른 피드백을 가한다.
        user: UserEntity = self.user_dict[inst.user_id]
        item: ItemEntity = self.item_dict[inst.item_id]

        uidx = self.user_id_to_idx[user.user_id]
        for x_name in user.top_n_decision_tags_set:
            if not x_name in self.tag_name_to_idx:
                continue
            x_idx = self.tag_name_to_idx[x_name]

            for y_name in item.tags_set:
                if not y_name in self.tag_name_to_idx:
                    continue
                y_idx = self.tag_name_to_idx[y_name]
                weight = self.arr_user_idx_to_weights[uidx][x_idx][y_idx]
                self.arr_user_idx_to_weights[uidx][x_idx][y_idx] = weight + (
                    _feedback * self.arr_tags_score[x_idx][y_idx] * weight
                )
            # end : for (item.tags)
        # end : for (user.top_n_tags)

        # 오차정도 반환
        return error_rate

    # end : protected void adjust()

    @property
    def model(self) -> CorrelationModel:
        return self._model

    @property
    def arr_tags_score(self) -> np.ndarray:
        """>>> `TagsScore[T, T]`"""
        return self.model.arr_tags_score

    @property
    def _user_based_tags_pccay(self) -> np.ndarray:
        """>>> `UsersPCC[T, T]`"""
        return self.model._ub_pcc_co_occur_score

    @property
    def _item_based_pcc_tags_array(self) -> np.ndarray:
        """>>> `ItemsPCC[T, T]`"""
        return self.model._ib_pcc_co_occur_score


# end : class
