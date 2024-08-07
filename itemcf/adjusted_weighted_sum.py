import math

import numpy as np

from core import BaseAction, BaseEstimator, UserEntity  # , ItemEntity
from .base_distance_model import BaseDistanceModel


class AdjustedWeightedSum(BaseEstimator):

    def __init__(
        self,
        model: BaseDistanceModel,
    ) -> None:
        super().__init__(model)
        # if not self.has_similarity_array:
        #    self.model.similarity()
        # end : if (computed_similarities?)

    # end : init()

    def _estimate_(self, inst: BaseAction) -> BaseAction:
        """보정된 가중치 합을 통한 의사결정 점수 추정 (항목기반)"""
        denum = numer = score = 0.0
        if not inst.item_id in self.item_id_to_idx:
            return inst
        x_idx: int = self.item_id_to_idx[inst.item_id]
        user: UserEntity = self.user_dict[inst.user_id]

        for item_id in user.dict_of_decision_item_ids["view"]:
            if not item_id in self.item_id_to_idx:
                continue
            y_idx: int = self.item_id_to_idx[item_id]
            score = self.arr_similarities[x_idx, y_idx]
            numer += score * (1 - self.arr_idx_to_mean_freq[y_idx])
            # numer += score - self.arr_idx_to_mean_freq[y_idx]
            denum += math.fabs(score)
        # end : for (items)

        # !Default Voting
        inst.estimated_score = (
            0.0 if denum == 0.0 else self.arr_idx_to_mean_freq[x_idx] + (numer / denum)
        )
        # numer = 0.0 if denum == 0.0 else numer / denum
        # inst.estimated_score = self.arr_idx_to_mean_freq[x_idx] + numer
        return inst

    # end : protected override BaseAction estimate()

    @property
    def model(self) -> BaseDistanceModel:
        return self._model

    """
    @property
    def has_similarity_array(self) -> bool:
        return self.model.__computed_similarities
    """

    @property
    def arr_similarities(self) -> np.ndarray:
        """2D similarity matrix"""
        return self.model.arr_similarties

    @property
    def arr_idx_to_mean_freq(self) -> np.ndarray:
        """1D mean Freq. matrix"""
        return self.model.idx_to_mean_freq_array

    @staticmethod
    def create_models_parameters() -> dict:
        """
        사용하지 않습니다.
        ====
        """
        pass

    def _set_model_params_(
        self,
        model_params: dict,
    ) -> None:
        """
        사용하지 않습니다.
        ====
        """
        pass


# end : class
