from sklearn.decomposition import NMF

from core import BaseDataSet
from .decomposition_model import DecompositionModel


class NMFDecompositionModel(DecompositionModel):
    """
    - 요약:
        - NMF로 latent factors를 채웁니다.
        - estimator 모듈을 호출하기 전에, analysis함수를 실행하세요.

    >>>  model.analysis()
    """

    def __init__(
        self,
        dataset: BaseDataSet,
        model_params: dict,
        # factors_dim: int,
        # learning_rate: float = 0.1,
        # generalization: float = 0.5,
    ) -> None:
        # super().__init__(dataset, factors_dim)
        self.__is_preprocessed = False
        super(DecompositionModel, self).__init__(dataset, model_params)

    # end : init()
    @staticmethod
    def create_models_parameters(
        factors_dim: int,
        factorizer_iters: int = 200,
    ) -> dict:
        return {
            "factors_dim": factors_dim,
            "factorizer_iters": factorizer_iters,
        }

    # end : public static override dict create_models_parameters()

    def _factorizer_def_(self) -> None:
        print(f"[{type(self).__name__}] factorizer allocation")
        self._factorizer = NMF(
            n_components=self._dimension,
            init="random",
            solver="cd",
            beta_loss="frobenius",
            max_iter=self._factorizer_iters,
        )

    # end : protected override void factorizer_def()

    def _process_(self) -> None:
        if self.__is_preprocessed:
            return
        print(f"[{type(self).__name__}] decomposition")
        self.users_factors = self.factorizer.fit_transform(X=self.arr_dicisions)
        self.items_factors = self.factorizer.components_
        self.__is_preprocessed = True

    # end : protected override void process()

    def _postprocess_(self) -> None:
        # MF이므로, 후처리 없음
        pass

    # end : protected override void _postprocess_()

    @property
    def factorizer(self) -> NMF:
        return self._factorizer


# end : class
