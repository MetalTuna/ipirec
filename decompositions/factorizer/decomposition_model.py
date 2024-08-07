from abc import *

import numpy as np

from core import BaseModel, BaseDataSet, BaseAction


class DecompositionModel(BaseModel):
    """
    - 요약:
        - 행렬분해 모델을 구성하기 위한 추상클래스입니다.
        - 잠재특성의 학습을 통일하기 위해 만들어진 모듈입니다.
    - 추상함수:
        - protected void factorizer_def()
        - protected void process()
        - protected void preprocess()
    """

    def __init__(
        self,
        dataset: BaseDataSet,
        model_params: dict,
        # factors_dim: int,
        # learning_rate: float = 0.1,
        # generalization: float = 0.5,
    ) -> None:
        super().__init__(dataset, model_params)

        ## member_vars_define
        # self._dimension = factors_dim
        self._dimension = int()
        """
        - 요약:
            - 축소할 차원의 크기(d)를 정수로 입력합니다.
        """
        self._factorizer_iters = int()
        """
        - 요약:
            - 분해된 행렬을 보정하는 반복 횟수입니다. 
        """
        self.arr_dicisions: np.ndarray = None
        """
        - 요약:
            - 사용자들(m)의 항목들(n)에 대한 의사결정 행렬입니다.
            - 의사결정은 0과 1로 구성됩니다.
        """
        self.users_factors: np.ndarray = None
        """
        - 요약:
            - 사용자들에 대한 잠재특성 행렬입니다.
            - 행렬의 크기는 사용자들의 수(m) X 축소된 차원의 수(d)를 갖습니다.
        """
        self.items_factors: np.ndarray = None
        """
        - 요약:
            - 항목들에 대한 잠재특성 행렬입니다.
            - 행렬의 크기는 항목들의 수(n) X 축소된 차원의 수(d)를 갖습니다.
        """
        self._factorizer = None
        """
        - 요약:
            - 의사결정 행렬을 분해하는 모듈입니다.
        """
        self._set_model_params_(model_params=model_params)

    # end : init()

    @staticmethod
    def create_models_parameters(
        factors_dim: int,
        factorizer_iters: int = 5,
    ) -> dict:
        return {
            "factors_dim": factors_dim,
            "factorizer_iters": factorizer_iters,
        }

    # end : public static override dict create_models_parameters()

    def _set_model_params_(
        self,
        model_params: dict,
    ) -> None:
        kwd = "factors_dim"
        if not kwd in model_params:
            raise KeyError()
        self._dimension: int = model_params[kwd]
        kwd = "factorizer_iters"
        if not kwd in model_params:
            raise KeyError()
        self._factorizer_iters: int = model_params[kwd]

    # end : protected override void set_model_params()

    @abstractmethod
    def _factorizer_def_(self) -> None:
        """
        - 요약:
            - decomposition을 위한 factorizer instance를 정의해 할당합니다.
            - self._factorizer를 할당하세요.
        - 주의:
            - 메타데이터는 반드시 적재돼야 합니다. (향후, 필요에 따라 BaseDataSet에 관련변수 하나 추가해서 처리)
            - 의사결정 행렬은 옵션
        """
        raise NotImplementedError()

    # end : protected abstract void factorizer_def()

    def __decision_matrix_allocation__(
        self,
        decisions_collection: list = None,
    ) -> None:
        """
        - 요약:
            - 의사결정 행렬을 구성합니다.
        - 매개변수:
            - decisions_collection (list, optional):
                - 의사결정 행렬을 구성할 의사결정 목록입니다.
                - None이면, 봤다에 대한 의사결정 행렬이 생성됩니다.
        """

        print("[DecompositionModel] decisions matrix allocation")
        decisions_list: list = (
            self.view_list if decisions_collection == None else decisions_collection
        )

        # 의사결정 행렬 만들기
        self.arr_dicisions = np.zeros(
            shape=(
                self.users_count,
                self.items_count,
            ),
            dtype=np.short,
        )

        for inst in decisions_list:
            inst: BaseAction
            if not inst.user_id in self.user_id_to_idx:
                continue
            uidx: int = self.user_id_to_idx[inst.user_id]
            if not inst.item_id in self.item_id_to_idx:
                continue
            iidx: int = self.item_id_to_idx[inst.item_id]
            self.arr_dicisions[uidx][iidx] = 1
        # end : for (decisions)

    # end : private void decision_matrix_allocation()

    def _preprocess_(self) -> None:
        """
        - 요약:
            - factorizer instance 할당 (전처리)
        """
        self._factorizer_def_()
        self.__decision_matrix_allocation__()

    # end : protected void preprocess()

    def _process_(self) -> None:
        """
        - 요약:
            - 의사결정 행렬을 분해하는 과정을 여기에서 다룹시다.
        """
        raise NotImplementedError()

    # end : protected void process()

    def _postprocess_(self) -> None:
        """
        - 요약:
            - MF를 기준으로 optimization하므로, SVD관련 클래스들은 여기에서 US를 P로 합칩시다.
        """
        raise NotImplementedError()

    # end : protected void postprocess()


# end : class
