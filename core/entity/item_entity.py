from abc import *
from .base_entity import BaseEntity


class ItemEntity(BaseEntity):
    """
    - 요약:
        - 항목을 다루기 위한 추상클래스 입니다.
    """

    def __init__(
        self,
        item_id: int = -1,
        tags_set: set = set(),
    ):
        super().__init__()
        self.item_id = item_id
        """항목 번호 (-1이면 해당 없음)"""
        self.item_name = str()
        """항목 이름"""
        self.tags_set = tags_set
        """항목이 속하는 태그들의 집합"""
        self.set_of_ips = set()
        """항목이 속하는 라이센스들의 집합(영화는 IP들의 집합)"""
        self.set_of_categories = set()
        """항목이 속하는 카테고리들의 집합(영화는 장르들의 집합)"""
        self.dict_of_users_decision = {
            "view": set(),
            "like": set(),
            "purchase": set(),
        }
        """
        - 요약: 
            - 이 항목에 의사결정한 사용자들의 사전 
        - 구조
            - Key: keyword (str) - ["view", "like", "purchse"]
            - Value: user_ids (set)
        - 구성
        "view", "like", "purchse": user_ids (set)
        """

    # end : init()

    def relations_clear(self) -> None:
        """의사결정 내역에 따른 관계표현에 대한 변수들만 초기화 합니다."""
        for k in self.dict_of_users_decision.keys():
            self.dict_of_users_decision[k].clear()

    # end : public void clear()

    @property
    def set_of_view_user_ids(self) -> set:
        """항목을 조회한 사용자들의 집합"""
        return self.dict_of_users_decision["view"]

    @property
    def set_of_like_user_ids(self) -> set:
        """항목에 좋아요한 사용자들의 집합"""
        return self.dict_of_users_decision["like"]

    @property
    def set_of_purchase_user_ids(self) -> set:
        """항목을 구입한 사용자들의 집합"""
        return self.dict_of_users_decision["purchse"]


# end : class
