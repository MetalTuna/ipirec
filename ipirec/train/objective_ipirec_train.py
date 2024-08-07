## builtin
from configparser import ConfigParser

## 3rd Pty.
import numpy as np

## Custom LIB.
from core import (
    BaseObjectiveScore,
    BaseObjectiveTrain,
    BaseTrain,
    DecisionType,
    BaseAction,
    BaseModel,
)

from ..model.correlation_model import CorrelationModel
from ..model.base_corr_estimator import BaseCorrelationEstimator


class ObjectiveIPIRecTrain(BaseObjectiveTrain):

    def __init__(
        self,
        trainable_estimator: BaseTrain,
        objective_inst: BaseObjectiveScore,
    ) -> None:
        super().__init__(
            trainable_estimator,
            objective_inst,
        )

    def _preprocess_(self):
        return super()._preprocess_()

    def _process_(self):
        return super()._process_()

    def _postprocess_(self):
        return super()._postprocess_()

    ### properties
    @property
    def model(self) -> CorrelationModel:
        return self._model

    @property
    def estimator(self) -> BaseCorrelationEstimator:
        return self._estimator

    @property
    def user_dict(self) -> dict:
        return self._dataset.user_dict

    @property
    def user_id_to_idx(self) -> dict:
        return self._dataset.user_id_to_idx

    @property
    def item_dict(self) -> dict:
        return self._dataset.item_dict

    @property
    def item_id_to_idx(self) -> dict:
        return self._dataset.item_id_to_idx

    @property
    def tags_dict(self) -> dict:
        return self._dataset.tags_dict

    @property
    def tag_name_to_idx(self) -> dict:
        return self._dataset.tag_name_to_idx

    @property
    def users_count(self) -> int:
        return self.users_count

    @property
    def items_count(self) -> int:
        return self.items_count

    @property
    def tags_count(self) -> int:
        return self.tags_count

    @property
    def view_list(self) -> list:
        return self._dataset.view_list

    @property
    def like_list(self) -> list:
        return self._dataset.like_list

    @property
    def purchase_list(self) -> list:
        return self._dataset.purchase_list

    @property
    def arr_tags_score(self) -> np.ndarray:
        return self.estimator.arr_tags_score

    @property
    def arr_user_idx_to_weights(self) -> np.ndarray:
        return self.estimator.arr_user_idx_to_weights

    @property
    def config_info(self) -> ConfigParser:
        return self.model._config_info


# end : class
