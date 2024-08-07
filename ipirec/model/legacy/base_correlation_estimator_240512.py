"""
[IPIRec] Legacy 
- 수정일: 2024.05.12.

- 요약:
    - 사용하지 않습니다.
    - 작업흐름을 남기기 위해 보존된 모듈입니다.
        - 필요에 따라 보관위치를 변경하세요.
"""

import math
import numpy as np

from core import *
from .correlation_model import CorrelationModel
from ..defines.ipirec_weight_option import IPIRecWeightOption


class BaseCorrelationEstimator(BaseTrain):

    def __init__(
        self,
        model: BaseModel,
        model_params: dict,
    ) -> None:
        if not isinstance(model, CorrelationModel):
            raise TypeError()
        super().__init__(model, model_params)

    # end : init()

    # end : protected void set_model_params()

    def __fit__(self, target_decision: DecisionType, n: int) -> float:
        return super().__fit__(target_decision, n)

    # end : private override float fit()

    def _preprocess_(self):
        for _ in range(self._score_iterations):
            self.__adjust_tags_score__()
        # end : for (score_iter)

    def __adjust_tags_score__(self) -> None:
        self.arr_user_idx_to_weights[0] = np.ones(
            shape=(
                self.tags_count,
                self.tags_count,
            )
        )
        weight_option = IPIRecWeightOption.E_GLOBAL
        for inst in self.view_list:
            inst: BaseAction
            user: UserEntity = self.user_dict[inst.user_id]
            item: ItemEntity = self.item_dict[inst.item_id]

            # estimate
            estimated_score = self.__weighted_sum__(
                user_id=user.user_id,
                item_id=item.item_id,
                weight_option=weight_option,
            )
            # is fit?
            if estimated_score == 1.0:
                continue
            # feedback
            error_rate = 1.0 - estimated_score
            update_rate = self._score_learning_rate * error_rate

            """
            # NMF와 같이 채울 수는 없을까?
            weight_norm = score_norm = 0.0
            for x_name in user.top_n_decision_tags_set:
                if not x_name in self.tag_name_to_idx:
                    continue
                x_idx: int = self.tag_name_to_idx[x_name]
                for y_name in item.tags_set:
                    if not y_name in self.tag_name_to_idx:
                        continue
                    y_idx: int = self.tag_name_to_idx[y_name]
                    weight_norm += math.fabs(
                        self.arr_user_idx_to_weights[0][x_idx][y_idx]
                    )
                    score_norm += math.fabs(self.arr_tags_score[x_idx][y_idx])
                # end : for (T(i))
            # end : for (NT(u))
            """

            for x_name in user.top_n_decision_tags_set:
                if not x_name in self.tag_name_to_idx:
                    continue
                x_idx: int = self.tag_name_to_idx[x_name]
                for y_name in item.tags_set:
                    if x_name is y_name:
                        continue
                    if not y_name in self.tag_name_to_idx:
                        continue
                    y_idx: int = self.tag_name_to_idx[y_name]
                    corr: float = self.arr_tags_score[x_idx][y_idx]
                    weight: float = self.arr_user_idx_to_weights[0][x_idx][y_idx]
                    weight = math.tanh(weight + (corr * weight * update_rate))

                    """
                    # NMF
                    weight = weight * (
                        (1.0 + (corr * self._score_learning_rate))
                        / (estimated_score + (weight_norm * self._score_generalization))
                    )
                    """
                # end : for (T(i))
            # end : for (NT(u))
        # end : for (decisions)

        self.model.arr_tags_score = np.multiply(
            self.arr_tags_score,
            self.arr_user_idx_to_weights[0],
        )

    # end : private void adjust_tags_score()

    def _estimate_(self, inst: BaseAction) -> BaseAction:
        inst.estimated_score = self.__weighted_sum__(
            user_id=inst.user_id,
            item_id=inst.item_id,
        )
        return inst

    # end : protected override BaseAction estimate()

    def _adjust_(self, inst: BaseAction) -> float:
        return super()._adjust_(inst)

    def __weighted_sum__(
        self,
        user_id: int,
        item_id: int,
        weight_option: IPIRecWeightOption = IPIRecWeightOption.E_USER,
    ) -> float:
        user: UserEntity = self.user_dict[user_id]
        item: ItemEntity = self.item_dict[item_id]

        # 태그점수 보정이면 0, 개인화된 학습이면 user_idx가 주어집니다.
        uidx = (
            self.user_id_to_idx[user_id]
            if weight_option == IPIRecWeightOption.E_USER
            else 0
        )
        numer = denom = 0.0
        for x_name in user.top_n_decision_tags_set:
            if not x_name in self.tag_name_to_idx:
                continue
            x_idx: int = self.tag_name_to_idx[x_name]
            for y_name in item.tags_set:
                if x_name is y_name:
                    continue
                if not y_name in self.tag_name_to_idx:
                    continue
                y_idx: int = self.tag_name_to_idx[y_name]

                corr = self.arr_tags_score[x_idx][y_idx]
                weight = self.arr_user_idx_to_weights[uidx][x_idx][y_idx]
                weighted_score = corr * weight
                condition_score = (
                    1.0
                    if y_name in user.top_n_decision_tags_set
                    else self._default_voting_score
                )
                numer += weighted_score * condition_score
                denom += math.fabs(weighted_score)
            # end : for (T(i))
        # end : for (NT(u))

        return 0.0 if denom == 0.0 else numer / denom

    # end : private float weighted_sum()

    @staticmethod
    def create_models_parameters(
        top_n_decision_freq: int,
        co_occur_items_freq: int,
        score_iterations: int,
        score_learning_rate: float,
        score_generalization: float,
        weight_iterations: int,
        weight_learning_rate: float,
        weight_generalization: float,
        frob_norm_dist: int = 1,
        default_voting_score: float = 0.0,
    ) -> dict:
        # model_params = BaseTrain.create_models_parameters(learning_rate, generalization)
        return {
            "top_n_decision_freq": top_n_decision_freq,
            "co_occur_items_freq": co_occur_items_freq,
            "score_iterations": score_iterations,
            "score_learning_rate": score_learning_rate,
            "score_generalization": score_generalization,
            "weight_iterations": weight_iterations,
            "weight_learning_rate": weight_learning_rate,
            "weight_generalization": weight_generalization,
            "frob_norm_dist": frob_norm_dist,
            "default_voting_score": default_voting_score,
        }

    # public static override dict create_models_parameters()

    def _set_model_params_(self, model_params: dict) -> None:
        self._top_n_tags: int = model_params["top_n_decision_freq"]
        self._co_occur_items: int = model_params["co_occur_items_freq"]
        self._score_iterations: int = model_params["score_iterations"]
        self._score_learning_rate: float = model_params["score_learning_rate"]
        self._score_generalization: float = model_params["score_generalization"]
        self._weight_iterations: int = model_params["weight_iterations"]
        self.LEARNING_RATE: float = model_params["weight_learning_rate"]
        """weight learning rate"""
        self.REGULARIZATION: float = model_params["weight_generalization"]
        """weight generalization"""
        self._frob_norm: int = model_params["frob_norm_dist"]
        self._default_voting_score: float = model_params["default_voting_score"]

    @property
    def model(self) -> CorrelationModel:
        return self._model

    @property
    def arr_tags_score(self) -> np.ndarray:
        return self.model.arr_tags_score


# end : class
