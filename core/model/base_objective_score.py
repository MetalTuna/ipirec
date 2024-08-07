## built-in
from abc import *

## Custom
from .base_model import BaseModel
from .base_estimator import BaseEstimator


class BaseObjectiveScore(metaclass=ABCMeta):

    def __init__(
        self,
        estimator: BaseEstimator,
    ) -> None:
        self._estimator = estimator

    # end : init()

    ## abstract methods
    @abstractmethod
    def _tags_score_cost_(
        self,
        decisions_list: list,
    ) -> float:
        """태그점수 보정의 목적함수 계산기능 구현을 위한 추상함수입니다."""
        raise NotImplementedError()

    @abstractmethod
    def _personalization_cost_(
        self,
        decisions_list: list,
    ) -> float:
        """개인화 학습의 목적함수 계산기능 구현을 위한 추상함수입니다."""
        raise NotImplementedError()

    ## properties
    @property
    def _model(self) -> BaseModel:
        return self._estimator._model


# end : class
