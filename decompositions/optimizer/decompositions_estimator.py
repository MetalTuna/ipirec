import numpy as np

from core import BaseModel, BaseTrain, BaseAction
from core.defines.decision_type import DecisionType
from ..factorizer import DecompositionModel


class DecompositionsEstimator(BaseTrain):
    """
    - 요약:
        - 행렬분해 모델을 구성하기 위한 추상클래스입니다.
        - 잠재특성의 학습을 통일하기 위해 만들어진 모듈입니다.
            - 학습정책을 변경한다면 adjust()를 재정의하세요.
            - 경사하강법으로 잠재특성을 학습합니다. -- ALS, NMF
            - 관련문서 (노션):
                https://www.notion.so/colleykr/24-04-17-65deb88b71ed4072a9e94e660050505f?pvs=4
    """

    def __init__(
        self,
        model: BaseModel,
        model_params: dict,
        # learning_rate: float = 0.1,
        # generalization: float = 0.5,
    ) -> None:
        # 아빠가 행렬분해모델이냐
        if not isinstance(model, DecompositionModel):
            raise TypeError()
        self._frob_norm: int = 1
        self._train_iters: int = 150
        self._train_seq: list = None
        super().__init__(model, model_params)

    # end : init()

    def _estimate_(self, inst: BaseAction) -> BaseAction:
        uidx: int = self.user_id_to_idx[inst.user_id]
        iidx: int = self.item_id_to_idx[inst.item_id]

        estimated_score = 0.0
        user_factors = self.users_factors[uidx, :]
        item_factors = self.items_factors[:, iidx]
        estimated_score = np.matmul(user_factors, item_factors)
        inst.estimated_score = estimated_score
        return inst

    # end : protected override void estimate()

    @staticmethod
    def create_models_parameters(
        learning_rate: float = 0.1,
        generalization: float = 0.5,
        train_iters: int = 150,
        train_decision_types_seq: list = [d for d in DecisionType],
        frob_norm: int = 1,
    ) -> dict:
        _estimator_params = BaseTrain.create_models_parameters(
            learning_rate,
            generalization,
        )
        _estimator_params.update({"frob_norm": frob_norm})
        _estimator_params.update({"train_iters": train_iters})
        _estimator_params.update(
            {"train_seq": DecisionType.list_to_kwd_str(train_decision_types_seq)}
        )
        return _estimator_params

    # end : public static override dict create_models_parameters()

    def _set_model_params_(self, model_params: dict) -> None:
        BaseTrain._set_model_params_(self, model_params)
        self._frob_norm = int(model_params["frob_norm"])
        self._train_iters = int(model_params["train_iters"])
        self._train_seq = list()
        for s in model_params["train_seq"]:
            match (s):
                case "v":
                    self._train_seq.append(DecisionType.E_VIEW)
                case "l":
                    self._train_seq.append(DecisionType.E_LIKE)
                case "p":
                    self._train_seq.append(DecisionType.E_PURCHASE)
                case _:
                    raise ValueError()
        # end : for (train_seq)

    # end : protected void set_model_params()

    def train(
        self,
    ) -> None:
        for decision_type in self._train_seq:
            BaseTrain.train(
                self,
                target_decision=decision_type,
                n=self._frob_norm,
                emit_iter_condition=self._train_iters,
            )
        # end : for (decision_types)

    # public override void train()

    def __member_vars_allocation__(self) -> None:
        """사용하지 않음"""
        pass

    # end : private override void member_vars_allocation()

    def _preprocess_(self):
        # decisions matrix decomposition
        self.model._preprocess_()

    # end : protected override void preprocess()

    def _adjust_(self, inst: BaseAction) -> float:
        """
        - 요약:
            - 오차정도에 대한 특성 값들을 보정하고, 오차정도를 반환합니다.(목적함수에서 사용)

        - 매개변수:
            inst (BaseAction): 예측결과가 반영된 의사결정 인스턴스의 원소입니다.

        - 반환:
            float: 예측오차를 반환합니다. 목적함수에서 사용됩니다.
        """
        # __update_rate = self.LEARNING_RATE * error_rate
        uidx: int = self.user_id_to_idx[inst.user_id]
        iidx: int = self.item_id_to_idx[inst.item_id]
        user_factors: np.ndarray = self.users_factors[uidx, :]
        item_factors: np.ndarray = self.items_factors[:, iidx]
        update_factor = 0.0
        error_rate = 1.0 - inst.estimated_score

        for idx in range(self.factors_len):
            # update: user_factor
            update_factor = self.LEARNING_RATE * (
                (error_rate * item_factors[idx])
                - (self.REGULARIZATION * user_factors[idx])
            )
            self.users_factors[uidx, idx] += update_factor

            # update: item_factor
            update_factor = self.LEARNING_RATE * (
                (error_rate * user_factors[idx])
                - (self.REGULARIZATION * item_factors[idx])
            )
            self.items_factors[idx, iidx] += update_factor
        # end : for (factors)

        return error_rate

    # end : protected override float adjust()

    ### member properties
    @property
    def model(self) -> DecompositionModel:
        return self._model

    @property
    def factors_len(self) -> int:
        return self.model._dimension

    @property
    def users_factors(self) -> np.ndarray:
        return self.model.users_factors

    @property
    def items_factors(self) -> np.ndarray:
        return self.model.items_factors


# end : class
