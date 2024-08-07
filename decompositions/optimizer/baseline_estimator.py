import numpy as np

from core import BaseModel, BaseAction, UserEntity, ItemEntity
from .decompositions_estimator import DecompositionsEstimator


class BaselineEstimator(DecompositionsEstimator):

    def __init__(
        self,
        model: BaseModel,
        model_params: dict,
        # learning_rate: float = 0.1,
        # generalization: float = 0.5,
    ) -> None:
        # super().__init__(model, learning_rate, generalization)
        super(DecompositionsEstimator, self).__init__(model, model_params)
        self._mean_ratio_of_decisions = 0.0
        """\\mu = |R| / (|U| * |I|)"""
        self._user_idx_to_mean_freq: np.ndarray = None
        """b(u) = \\mu(u) - \\mu"""
        self._item_idx_to_mean_freq: np.ndarray = None
        """b(i) = \\mu(i) - \\mu"""
        self.__preprocess__()

    # end : init()

    def __preprocess__(self) -> None:
        """baseline prediction을 위한 전처리"""
        users_len = len(self.user_id_to_idx.keys())
        """No. of knowned users"""
        items_len = len(self.item_id_to_idx.keys())
        """No. of knowned items"""
        self._user_idx_to_mean_freq = np.zeros(
            shape=(users_len, 1),
            dtype=np.float32,
        )
        self._item_idx_to_mean_freq = np.zeros(
            shape=(items_len, 1),
            dtype=np.float32,
        )

        # 모든 의사결정 수의 평균 (sparsity)
        self._mean_ratio_of_decisions = len(self.view_list) / (users_len * items_len)

        # 각 사용자별 의사결정 수의 경향성
        # 사용자의 평균 의사결정 수 - 모든 의사결정 수의 평균
        for idx in range(users_len):
            user_id: int = self.user_idx_to_id[idx]
            user: UserEntity = self.user_dict[user_id]
            self._user_idx_to_mean_freq[idx] = (
                len(user.dict_of_decision_item_ids["view"]) / items_len
            ) - self._mean_ratio_of_decisions
        # end : for (users)

        # 각 항목별 의사결정 수의 경향성
        # 항목의 평균 의사결정 수 - 모든 의사결정 수의 평균
        for idx in range(items_len):
            item_id: int = self.item_idx_to_id[idx]
            item: ItemEntity = self.item_dict[item_id]
            self._item_idx_to_mean_freq[idx] = (
                len(item.dict_of_users_decision["view"]) / users_len
            ) - self._mean_ratio_of_decisions
        # end : for (items)

    # end : private void preprocess()

    def _estimate_(self, inst: BaseAction) -> BaseAction:
        inst = super()._estimate_(inst)
        # mu
        tendencies_score = self._mean_ratio_of_decisions
        # b(u)
        idx = self.user_id_to_idx[inst.user_id]
        tendencies_score += self._user_idx_to_mean_freq[idx]
        # b(i)
        idx = self.item_id_to_idx[inst.item_id]
        tendencies_score += self._item_idx_to_mean_freq[idx]

        # p(u,i) = mu + b(u) + b(i) + p(u)*q(i)
        inst.estimated_score += tendencies_score
        return inst

    # end : protected override BaseAction estimate()

    def _adjust_(self, inst: BaseAction) -> float:
        raise NotImplementedError()
        return super()._adjust_(inst)

    # end : protected override BaseAction adjust()


# end : class
