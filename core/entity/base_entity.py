from abc import *
import os


class BaseEntity(metaclass=ABCMeta):
    """
    - 요약:
        - 데이터 원소표현을 위한 추상클래스입니다.
    """

    def __init__(self):
        self._id: int = -1
        self._idx: int = -1

    # end : init()

    @property
    def get_id(self) -> int:
        return self._id

    @get_id.setter
    def id(self, id: int) -> None:
        if id < 0:
            raise ValueError()
        self._id = id

    @property
    def get_index(self) -> int:
        return self._idx

    @get_index.setter
    def idx(self, idx: int) -> None:
        if idx < 0:
            raise ValueError()
        self._idx = idx

    @staticmethod
    def load_collection(file_path: str) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError()
        if not os.path.isfile(file_path):
            raise IsADirectoryError()
        raise NotImplementedError()
        # return BaseEntity._load_entity_collection_(file_path)

    # @staticmethod
    @staticmethod
    def parse_entity(iter):
        """
        요약:
            pd.itertuples -> instance (BaseEntity)

        Args:
            iter (pd.iterrows): The element of column families (csv)

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError()


# end : class
