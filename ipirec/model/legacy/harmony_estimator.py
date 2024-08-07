import math

# 3rd party
import numpy as np

# users
from core import (
    BaseAction,
    BaseEstimator,
    BaseModel,
    UserEntity,
    ItemEntity,
    BaseModel,
    TagEntity,
)

# from ..model import CorrelationModel
from .correlation_model import CorrelationModel


class HarmonyEstimator(BaseEstimator):
    """
    사용하지 않음
    =====
    ---
    요약:
        의사결정에 따른 항목의 태그조합의 적합성을 분석합니다.
    """

    def __init__(
        self,
        model: BaseModel,
    ) -> None:
        super().__init__(model)

    def _estimate_(self, inst: BaseAction) -> BaseAction:
        """
        요약:
            기대점수 추정

        Args:
            inst (BaseAction): 실제 값

        Returns:
            BaseAction: 예측 값을 채운 것
        """
        user: UserEntity = self.user_dict[inst.user_id]
        item: ItemEntity = self.item_dict[inst.item_id]
        # res = BaseAction(user.user_id, item.item_id)

        decision_freq = len(user.dict_of_interaction_tags["view"])
        cnt = 0
        numer, denom = 0.0, 0.0
        ### 의사결정 수로 선호정도를 구해 차등할 때 사용
        opened_item_ids = user.dict_of_decision_item_ids["view"]
        for x_name in user.top_n_decision_tags_set:
            x_idx = self.tag_name_to_idx[x_name]
            x_inst: TagEntity = self.tags_dict[x_name]
            for y_name in item.tags_set:
                y_idx = self.tag_name_to_idx[y_name]
                y_inst: TagEntity = self.tags_dict[y_name]
                corr = self.tags_corr_array[x_idx][y_idx]
                # I(x,y) = I(x).intersection(I(y))
                co_occur_item_ids: set = x_inst.item_ids_set.intersection(
                    y_inst.item_ids_set
                )
                # item_ids = I(x,y).intersection(I(u))
                item_ids: set = opened_item_ids.intersection(co_occur_item_ids)
                # 항목의 열람 가능성
                decision_ratio = len(item_ids) / decision_freq
                # 태그관계
                numer += decision_ratio * corr
                denom += math.fabs(corr)
                cnt += 1
            # end : for (T(i))
        # end : for (NT(u))

        # 항목을 열람할 때, 태그관계에 의한 기대점수
        inst.estimated_score = 0.0 if cnt == 0 else numer / denom

        return inst

    @property
    def __model(self) -> CorrelationModel:
        return self._model

    @property
    def tags_corr_array(self) -> np.ndarray:
        return self.__model.arr_tags_score


# end : class
