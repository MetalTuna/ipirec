from abc import *

import numpy as np
from tqdm import tqdm

from core import BaseDataSet, BaseModel


class BaseDistanceModel(BaseModel):
    """
    - 요약:
        - 유사도 기반의 의사결정 예측모델 구성을 위한 추상클래스입니다.
    - 추상함수:

    """

    def __init__(
        self,
        dataset: BaseDataSet,
    ) -> None:
        # self.__computed_similarities = False
        # """유사도 구했냐"""
        self.idx_to_mean_freq_array: np.ndarray = None
        """item_idx to users mean freq. array"""
        self.arr_similarties: np.ndarray = None
        """2D similariy array """

        super().__init__(dataset)
        self.idx_to_mean_freq_array = np.zeros(
            shape=(self.items_count, 1),
        )
        # """item_idx to users mean freq. array"""
        self.arr_similarties = np.zeros(
            shape=(self.items_count, self.items_count),
        )
        # """2D similariy array """

    # end : init()

    def similarity(self) -> None:
        """
        - 요약:
            - 항목 간의 유사도를 구합니다.
        """
        self._preprocess_()
        self._process_()
        self._postprocess_()
        # self.__computed_similarities = True

    # end : public void similarity()

    ### self.analysis() => preprocess, process, postprocess
    @abstractmethod
    def _preprocess_(self) -> None:
        """전처리: 유사도 계산 전에 필요한 작업을 구현하세요."""
        raise NotImplementedError()

    @abstractmethod
    def __distance__(
        self,
        item_x: int,
        item_y: int,
    ) -> None:
        """
        - 요약:
            - 두 항목에 대한 유사도 계산을 구현하세요.

        - 매개변수:
            - item_x (int): 항목 x의 ID
            - item_y (int): 항목 y의 ID
        """
        raise NotImplementedError()

    # end : private void distance()

    def _process_(self) -> None:
        # 유사도 계산
        item_ids_list = list(self.item_id_to_idx.keys())
        # for item_x in self.item_id_to_idx.keys():
        for item_x in tqdm(
            iterable=self.item_id_to_idx.keys(),
            desc="DistanceModel.process()",
            total=len(self.item_id_to_idx.keys()),
        ):
            idx = self.item_id_to_idx[item_x]
            self.arr_similarties[idx][idx] = 0.0
            item_ids_list.remove(item_x)
            for item_y in item_ids_list:
                self.__distance__(item_x, item_y)
            # end : for (I - x)
        # end : for (items)

    # end : protected override void process()

    @abstractmethod
    def _postprocess_(self) -> None:
        """후처리: 유사도 계산 후에 필요한 작업을 구현하세요."""
        raise NotImplementedError()

    @staticmethod
    def create_models_parameters() -> dict:
        """
        사용하지 않습니다.
        ====
        """
        pass

    def _set_model_params_(
        self,
        model_params: dict,
    ) -> None:
        """
        사용하지 않습니다.
        ====
        """
        pass


# end : class
