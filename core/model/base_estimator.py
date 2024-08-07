from abc import *
from configparser import ConfigParser

from tqdm import tqdm

from ..entity import BaseAction
from .base_model import BaseModel
from .base_dataset import BaseDataSet
from .base_model_params import BaseModelParameters
from ..io import InstanceIO


# class BaseEstimator(metaclass=ABCMeta):
class BaseEstimator(BaseModelParameters, InstanceIO):
    """
    - 요약:
        - 추정관련 기능구현을 위한 추상 클래스입니다.

    - 멤버함수:
        - public list estimation(): 의사결정 예측내역들을 재추정합니다.
        - public float predict(): 의사결정 추정값을 구합니다.
            - estimate().estimated_score가 반환됩니다.

    - 추상함수:
        - protected BaseAction estimate(): 의사결정 추정 값을 계산하는 함수입니다.
    """

    def __init__(
        self,
        model: BaseModel,
    ) -> None:
        self._model = model

    # end : init()

    def estimation(
        self,
        decision_list: list,
    ) -> list:
        """
        요약:
            의사결정 예측내역들을 재추론합니다.

        Args:
            decision_list (list): BaseActions로 구성된 의사결정 내역들
        """
        estimated_list = list()
        # for inst in decision_list:
        for inst in tqdm(
            iterable=decision_list,
            desc="BaseEstimator.estimation()",
            total=len(decision_list),
        ):
            estimated_list.append(self._estimate_(inst))
        # end : for (decisions)
        return estimated_list

    # end : public void estimation()

    @abstractmethod
    def _estimate_(self, inst: BaseAction) -> BaseAction:
        """기대값 구하기"""
        raise NotImplementedError()

    def predict(self, user_id: int, item_id: int) -> float:
        """예측항목이 여과됐다면, 0점을 반환합니다."""
        return (
            0.0
            if not item_id in self.item_id_to_idx
            else self._estimate_(
                BaseAction(
                    user_id=user_id,
                    item_id=item_id,
                )
            ).estimated_score
        )

    ### properties
    @property
    def _dataset(self) -> BaseDataSet:
        return self._model._dataset

    ## direct references properties
    @property
    def user_dict(self) -> dict:
        return self._model._dataset.user_dict

    @property
    def item_dict(self) -> dict:
        return self._model._dataset.item_dict

    @property
    def tags_dict(self) -> dict:
        return self._model._dataset.tags_dict

    @property
    def users_count(self) -> int:
        return self._model._dataset.users_count

    @property
    def items_count(self) -> int:
        return self._model._dataset.items_count

    @property
    def tags_count(self) -> int:
        return self._model._dataset.tags_count

    @property
    def view_list(self) -> list:
        return self._model._dataset.view_list

    @property
    def like_list(self) -> list:
        return self._model._dataset.like_list

    @property
    def purchase_list(self) -> list:
        return self._model._dataset.purchase_list

    # index reference properties
    @property
    def user_id_to_idx(self) -> dict:
        return self._model.user_id_to_idx

    @property
    def user_idx_to_id(self) -> dict:
        return self._model.user_idx_to_id

    @property
    def item_id_to_idx(self) -> dict:
        return self._model.item_id_to_idx

    @property
    def item_idx_to_id(self) -> dict:
        return self._model.item_idx_to_id

    @property
    def tag_name_to_idx(self) -> dict:
        return self._model._dataset.tag_name_to_idx

    @property
    def tag_idx_to_name(self) -> dict:
        return self._model._dataset.tag_idx_to_name

    @property
    def _config_info(self) -> ConfigParser:
        return self._model._config_info

    @_config_info.setter
    def _config_info(
        self,
        _conf: ConfigParser,
    ) -> None:
        self._dataset._config_info = _conf


# end : class
