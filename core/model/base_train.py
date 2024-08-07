"""
[수정일]
- 2024.06.20 17:09. 실험모듈과 통합됨
    - 모델 피드백 함수들의 반복문을 변경함
        - 임계 값보다 오차변화가 작으면 탈출하는 조건이 제거됨. (목적함수를 별도 설계해서 탈출하기 위함)
        - 의사결정 내역 수로 길이 정규화된 예측오차 반환으로 값 계산이 변경됨.
- 2024.05.04 11:31. train함수의 종료조건이 변경됨
    - 오차변화가 emit_diff_condition (20.0)보다 작다면, 반복회수와 무관하게 반복문을 종료합니다.
    - fit함수에서 반환하는 예측오차 값은 평균에서 총합으로 변경됐습니다.
- 2024.05.23 11:06. fit함수에서 예측 값이 0.0이면, 태그점수 높이도록 수정한 것이 미반영됐음. (수정됨)
"""

from abc import *
import math

from tqdm import tqdm
import numpy as np

from ..entity import BaseAction
from ..defines import DecisionType
from .base_model import BaseModel
from .base_estimator import BaseEstimator


class BaseTrain(BaseEstimator):
    """
    - 요약:
        - 모델의 예측오차에 대한 피드백하는 기능구현을 위한 추상 클래스입니다.

    - 부모클래스:
        - BaseEstimator를 상속합니다.

    - 멤버함수:
        - public list estimation(): 의사결정 예측내역들을 재추정합니다.
        - public float predict(): 의사결정 추정값을 구합니다.
            - estimate().estimated_score가 반환됩니다.
        - public void train(): 의사결정 집합단위로 태그점수에 따른 오차를 관측하는 기능을 구현합니다.
            - 재정의한다면, 모델의 목적함수와 피드백을 함께 정의하세요.
        - private void member_vars_allocation(): 멤버변수들을 할당합니다. 구조가 다른 모델을 사용한다면, 이 함수를 변경해 사용하세요.

    - 추상함수:
        - protected BaseAction estimate(): 의사결정 추정 값을 계산하는 함수입니다.
        - protected float adjust(): 태그점수 보정을 위한 함수입니다(의사결정 원소단위로 모델의 피드백을 가합니다).
            - 의사결정 원소단위로 호출돼고, 의사결정 오차를 반환합니다.
            - (실제 값 - 예측 값)이므로 음수가 반환될 수 있습니다.
    """

    def __init__(
        self,
        model: BaseModel,
        model_params: dict,
        # learning_rate: float = 0.1,
        # generalization: float = 0.5,
    ) -> None:
        super().__init__(model)
        self.EMIT_ITER_THRESHOLD = int()
        """탈출조건: 반복 횟수"""
        self.EMIT_ERROR_THRESHOLD = float()
        """탈출조건: 오차변화"""
        self.estimation_error_numer = 0.0
        """오차의 누산 (분자)"""
        self.estimation_error_denom = 0
        """추정된 항목의 수 (분모)"""
        self.LEARNING_RATE = float()
        """학습률"""
        self.REGULARIZATION = float()
        """일반화 인자"""
        self.arr_user_idx_to_weights: np.ndarray = None
        """np.ndarray"""

        self._current_mean_of_error_scores = float()

        self._set_model_params_(model_params=model_params)
        '''
        self.LEARNING_RATE = learning_rate
        """학습률"""
        self.REGULARIZATION = generalization
        """일반화 인자"""
        self.arr_user_idx_to_weights: np.ndarray = None
        '''
        self.__member_vars_allocation__()

        self._model._append_config_(
            model_params=model_params,
            section="Estimator",
            inst=self,
        )

    # end : init()

    @staticmethod
    def create_models_parameters(
        learning_rate: float = 0.1,
        generalization: float = 0.5,
    ) -> dict:
        return {
            "learning_rate": learning_rate,
            "generalization": generalization,
        }

    # end : public static override dict create_models_parameters()

    def _set_model_params_(self, model_params: dict) -> None:
        kwd = "learning_rate"
        if not kwd in model_params:
            raise KeyError()
        self.LEARNING_RATE: float = model_params[kwd]
        kwd = "generalization"
        if not kwd in model_params:
            raise KeyError()
        self.REGULARIZATION: float = model_params[kwd]
        """
        _sec = "Estimator"
        if not _sec in self._config_info:
            self._config_info.add_section(_sec)
            self._config_info.set(_sec, "name", type(self).__name__)
        for k, v in model_params.items():
            self._config_info.set(_sec, k, str(v))
        # end : for (params)
        """

    # end : protected override void set_model_params()

    def __member_vars_allocation__(self) -> None:
        """
        - 요약:
            - 멤버변수들을 할당합니다.
            - 구조가 다른 모델을 사용한다면, 이 함수를 변경해 사용하세요.
        """
        # array allocations
        if self.arr_user_idx_to_weights == None:
            try:
                self.arr_user_idx_to_weights: np.ndarray = np.ones(
                    shape=(
                        self.users_count,
                        self.tags_count,
                        self.tags_count,
                    ),
                    dtype=np.float32,
                )
            except Exception as e:
                print(e)
        # end : if (weight_arr_allocated?)

    # end : private void member_vars_allocation()

    def train(
        self,
        target_decision: DecisionType = DecisionType.E_VIEW,
        n: int = 1,
        emit_iter_condition: int = 10,
        emit_diff_condition: float = 20.0,
    ) -> None:
        """
        - 요약:
            - 적합한 태그점수 계산을 위한 함수: 실행단위는 의사결정 집합

        - 매개변수:
            - target_decision (DecisionType, optional): 의사결정 타입을 선택합니다. 기본 값은 봤다입니다.
            - n (int, optional): Frobenious norm의 n입니다. (기본 값은 1)
                - 유클리드 거리에 근거한 RMSE를 구하려면 n = 2로 하세요.
            - emit_iter_condition (int, optional): 반복횟수에 대한 학습종료 조건을 정합니다. (기본 값은 100)
            - emit_diff_condition (float, optional): 오차변화에 대한 학습종료 조건을 정합니다. (기본 값은 20.0)
        """
        self._preprocess_()

        self.EMIT_ITER_THRESHOLD = emit_iter_condition
        self.EMIT_ERROR_THRESHOLD = emit_diff_condition
        error_rate = 0.0
        error_list = list()
        error_list.append(error_rate)

        print(f"[{type(self).__name__}]\n")
        for _ in tqdm(
            iterable=range(self.EMIT_ITER_THRESHOLD),
            desc="estimator.train()",
            total=self.EMIT_ITER_THRESHOLD,
        ):
            error_rate = self.__fit__(target_decision, n)
            error_list.append(error_rate)

            _err_dist = np.fabs(error_list[_] - error_list[_ + 1])
            if _err_dist < emit_diff_condition:
                break
            # if math.fabs(error_rate - error_list[iterations - 1])  < self.EMIT_ERROR_THRESHOLD
            #     break
        # end : for (iterations_condition)
        # self.__train_errors_list = error_list

    # end : public void train()

    def __fit__(
        self,
        target_decision: DecisionType,
        n: int,
    ) -> float:
        """
        - 요약:
            - 학습데이터 단위로 의사결정 종류에 대한 추정 오차들을 집계합니다.

        - 매개변수:
            - target_decision (DecisionType): 추정할 의사결정 타입을 선택합니다.
            - n (int, optional): Frobenious norm의 n입니다. (기본 값은 1)
                - 유클리드 거리에 근거한 RMSE를 구하려면 n = 2로 하세요.

        - 예외:
            - ValueError: 정의되지 않은 DecisionType을 사용할 경우, 예외가 발생합니다.

        - 반환:
            - float: 오차의 비율을 반환합니다(현재 구조는 거리누산 값의 평균을 구합니다).
        """
        error_rate = 0.0
        decisions_list: list = self._dataset._decision_dict[
            DecisionType.to_str(target_decision)
        ]
        self.estimation_error_numer = 0.0
        self.estimation_error_denom = 0.0

        for iter in tqdm(
            iterable=decisions_list,
            desc=f"[{DecisionType.to_str(target_decision)}] Adjust",
            total=len(decisions_list),
            # leave=False,
        ):
            iter: BaseAction
            if not iter.user_id in self.user_id_to_idx:
                continue
            if not iter.item_id in self.item_id_to_idx:
                continue
            iter = self._estimate_(iter)

            ## [24.05.23] 코드 꼬였나봄 ;;;
            # 0.0이면 태그점수 높이도록 수정하도록 변경했음;;
            # if iter.estimated_score == 0.0 or iter.estimated_score == 1.0:
            if iter.estimated_score == 1.0:
                continue

            error_rate = self._adjust_(iter)
            self.estimation_error_numer += math.fabs(error_rate) ** n
            self.estimation_error_denom += 1.0
            # error_rate = self.__adjust__(iter)
            # error_dist += math.pow(error_rate, 2.0)
        # end : for (decisions)

        if self.estimation_error_denom == 0.0:
            error_rate = 0.0
            self._current_mean_of_error_scores = 0.0
        else:
            # [24.06.20 11:26] 평균오차 구하도록 수정
            error_rate = self.estimation_error_numer ** (1 / n)
            self._current_mean_of_error_scores = (
                self.estimation_error_numer / self.estimation_error_denom
            ) ** (1 / n)
        """
        error_rate = (
            0.0
            if self.estimation_error_denom == 0.0
            else self.estimation_error_numer ** (1 / n)
            # else (self.estimation_error_numer / self.estimation_error_denom) ** (1 / n)
        )
        """

        return error_rate

    # end : private float fit()

    @abstractmethod
    def _preprocess_(self):
        """
        - 요약:
            - 모델의 학습에 필요한 전처리 정의를 위한 추상함수 입니다.
            - BaseTrain.train()이 호출되면, 이 함수부터 실행됩니다.
        """
        raise NotImplementedError()

    @abstractmethod
    def _adjust_(
        self,
        inst: BaseAction,
    ) -> float:
        """
        - 요약:
            - 태그점수 보정을 위한 함수: 실행단위는 의사결정 원소
        """
        raise NotImplementedError()

    # end : protected float adjust()


# end : class
