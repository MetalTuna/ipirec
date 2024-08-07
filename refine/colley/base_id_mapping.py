from abc import *


class BaseIDMapping(metaclass=ABCMeta):
    """상품과 게시글을 통합된 항목으로 변환하기 위한 추상클래스입니다."""

    def __init__(self):
        self._src_id = -1
        self.item_id = -1
        self.name = ""
        self.tags = set()

    # public override Any init()
    def __init__(self, src_id: int, item_id: int, name: str):
        self._src_id = src_id
        self.item_id = item_id
        self.name = name
        self.tags = set()

    @abstractmethod
    def _set_record_(self, r) -> None:
        """The r is element of pd.iterrows()"""
        raise NotImplementedError()
