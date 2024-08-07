import numpy as np
from sklearn.decomposition import TruncatedSVD

from core import BaseDataSet
from .decomposition_model import DecompositionModel


class TruncatedSVDModel(DecompositionModel):

    def __init__(
        self,
        dataset: BaseDataSet,
        model_params: dict,
        # factors_dim: int,
        # factorizer_iters: int = 10,
    ) -> None:
        # self.__factorizer_iters = factorizer_iters
        self.singular_values: np.ndarray = None
        """
        - 요약:
            - 특이 값으로 채워진 행렬입니다.
                - 행렬의 크기는 축소된 차원의 수(d) X 축소된 차원의 수(d)를 갖습니다.
            - 인줄 알았는데, self._factorizer가 1차원(d, 1) 특이 값만 출력함(대각행렬만 선별된 목록이란 뜻임)
        """
        # super().__init__(dataset, factors_dim)
        super(DecompositionModel, self).__init__(dataset, model_params)

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

    def _factorizer_def_(self) -> None:
        self._factorizer = TruncatedSVD(
            n_components=self._dimension,
            n_iter=self._factorizer_iters,
            # algorithm="randomized",
            # n_iter=self.__factorizer_iters,
            # power_iteration_normalizer="LU",
        )

    def _process_(self) -> None:
        self._factorizer = self._factorizer.fit(X=self.arr_dicisions)
        self.users_factors = self._factorizer.fit_transform(X=self.arr_dicisions)
        self.singular_values = self._factorizer.singular_values_
        self.items_factors = self._factorizer.components_
        print()

    # end : protected override void process()

    def _postprocess_(self) -> None:
        self.users_factors = self.users_factors * self.singular_values
        # self.users_factors = np.matmul(self.users_factors, self.singular_values)

    @property
    def factorizer(self) -> TruncatedSVD:
        return self._factorizer


# end : class
