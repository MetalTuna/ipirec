import pandas as pd

from .base_entity import BaseEntity


class TagEntity(BaseEntity):
    """
    - 요약:
        - 태그 객체를 다루는 클래스 입니다.
    """

    def __init__(
        self,
        tag_id: int,
        tag_name: str,
        tag_idx: int = -1,
    ):
        super().__init__()
        self.id = tag_id
        """태그 식별 번호(DB에서의 태그 ID)"""
        self.tag_name = tag_name
        """태그 이름"""
        self._idx = tag_idx
        """태그 색인번호"""
        self.user_ids_set = set()
        """이 태그에 의사결정한 사용자들의 집합"""
        self.item_ids_set = set()
        """이 태그에 속하는 항목들의 집합"""

        self.decisions_freq_dict = {
            "view": 0,
            "like": 0,
            "purchase": 0,
            "total": 0,
        }
        """
        - 요약:
            - 이 태그에 의사결정된 빈도 수
        - 구성:
            - Key: view, like, purchase, total,
            - Value: No. of decisions (int)
        """

    # end : init()

    def relations_clear(self) -> None:
        """의사결정 관계표현에 사용된 변수들을 초기화합니다."""

        self.user_ids_set.clear()
        """이 태그에 의사결정한 사용자들의 집합"""
        self.item_ids_set.clear()
        """이 태그에 속하는 항목들의 집합"""

        for k in self.decisions_freq_dict.keys():
            self.decisions_freq_dict[k] = 0
        # end : for (decision_types)

    # end : public void relations_clear()

    def member_collections_init(self) -> None:
        """이 클래스 인스턴스의 멤버변수들을 초기화합니다."""
        self.user_ids_set.clear()
        self.item_ids_set.clear()
        self.decisions_freq_dict = {
            "view": 0,
            "like": 0,
            "purchase": 0,
            "total": 0,
        }

    # end : public void member_collections_init()

    @staticmethod
    def load_collection(file_path: str) -> dict:
        tags_dict = dict()
        for _, r in pd.read_csv(file_path).iterrows():
            # tag_id,tag
            inst = TagEntity.parse_entity(r)
            tags_dict.update({inst.tag_name: inst})
        # end : for (tags_list)
        return tags_dict

    # end : public static dict load_collection()

    @staticmethod
    def parse_entity(iter):
        tag_id = int(iter["tag_id"])
        tag_name = iter["tag"].strip()
        return TagEntity(tag_id, tag_name)

    # end : public static TagEntity parse_entity()
