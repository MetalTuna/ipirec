## built-in
from abc import *

## 3rd Pty.
from tqdm import tqdm

## Custom LIB.
from .base_dataset import BaseDataSet
from .base_model import BaseModel
from .base_estimator import BaseEstimator
from .base_train import BaseTrain
from .base_objective_score import BaseObjectiveScore


class BaseObjectiveTrain:

    def __init__(
        self,
        trainable_estimator: BaseTrain,
        objective_inst: BaseObjectiveScore,
    ) -> None:
        self._trainer = trainable_estimator
        """모델의 학습들이 정의된 클래스 인스턴스입니다."""
        self._objective_inst = objective_inst
        """목적함수 계산이 정의된 클래스 인스턴스입니다."""

    # end : init()

    def train(self):
        self._preprocess_()
        self._process_()
        self._postprocess_()

    # end : public Any train()

    @abstractmethod
    def _preprocess_(self):
        raise NotImplementedError()

    @abstractmethod
    def _process_(self):
        raise NotImplementedError()

    @abstractmethod
    def _postprocess_(self):
        raise NotImplementedError()

    # @abstractmethod
    # def _training_sequences_define_(self):
    #    raise NotImplementedError()

    ## properties
    @property
    def _model(self) -> BaseModel:
        return self._trainer._model

    @property
    def _dataset(self) -> BaseDataSet:
        return self._trainer._dataset

    @property
    def _estimator(self) -> BaseEstimator:
        return self._trainer


# end : class
