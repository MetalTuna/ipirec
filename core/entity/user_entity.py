import pandas as pd

from .base_entity import BaseEntity


class UserEntity(BaseEntity):
    """
    - 요약:
        - 사용자 객체를 다루는 클래스 입니다.
    """

    def __init__(
        self,
        user_idx: int = -1,
        user_id: int = -1,
    ):
        super().__init__()
        self._id = user_id
        """사용자 식별 번호"""
        self._idx = user_idx
        self.set_of_interest_tags = set()
        """관심등록한 태그들의 집합"""
        # self.decision_item_ids_dict = {
        self.dict_of_decision_item_ids = {
            "view": set(),
            "like": set(),
            "purchase": set(),
        }
        """
        - 요약: 
            - 이 사용자가 의사결정한 항목들의 사전
            - r(u,i) = |T| * |I| (항목에 |T|개의 태그가 있고, 각 태그들의 길이가 |I|임)
        - 구조
            - Key: keyword (str) - ["view", "like", "purchase"]
            - Value: item_ids (set)
        - 구성
        "view", "like", "purchase": {
            item_ids (set)
        }
        """
        self.dict_of_interaction_tags = {
            "view": set(),
            "like": set(),
            "purchase": set(),
        }
        """
        - 요약:
            - 이 사용자가 의사결정한 태그들의 사전
        - 구조
            - Key: keyword (str) - ["view", "like", "purchase"]
            - Value: tags_name (set)
        - 구성
        "view", "like", "purchase": {
            tags_name (set)
        }"""
        self.tags_decision_freq_dict = dict()
        """
        - 요약:
            - 태그별 의사결정(봤다만) 빈도 수: r(u,t) => |I(u)|
        - 구성:
            - Key: tag_name (str)
            - Value: decision_freq (int)
        """

        self.top_n_decision_tags_set = set()
        """
        - 요약:
            - Top-N 의사결정 태그들의 집합
        """

        self.candidate_item_ids_set = set()
        """
        - 요약:
            - 추천후보 상품목록 집합
        """
        self.estimated_items_score_list = list()
        """
        - 요약:
            - 항목에 대한 의사결정 추정점수 목록
        - 구성: instance (BaseAction)
        """
        self.recommended_items_dict = dict()
        """
        - 요약:
            - 추천항목 사전
        - 구성:
            - Key: item_id (int)
            - Value: predicted_score (float) 
        """

    # end : init()

    def relations_clear(self) -> None:
        """의사결정에 대한 관계표현에 사용된 멤버변수들을 초기화합니다."""

        for k in self.dict_of_decision_item_ids.keys():
            self.dict_of_decision_item_ids[k].clear()
        for k in self.dict_of_interaction_tags.keys():
            self.dict_of_interaction_tags[k].clear()
        self.tags_decision_freq_dict.clear()
        self.top_n_decision_tags_set.clear()
        self.candidate_item_ids_set.clear()
        self.estimated_items_score_list.clear()
        self.recommended_items_dict.clear()

    # end : public void relations_clear()

    @property
    def user_id(self) -> int:
        return self.id

    @staticmethod
    def load_collection(file_path: str) -> dict:
        """_summary_

        - Args:
            - file_path (str): user_list (csv)

        - Returns:
            - user_dict (dict): instance dictionary of users
                - Key: user_id (int)
                - Value: instance (UserEntity)
        """
        df = pd.read_csv(file_path)
        user_dict = dict()
        for _, r in df.iterrows():
            inst = UserEntity.parse_entity(r)
            inst.idx = _
            user_dict.update({inst.id: inst})
        # end : for (users)
        return user_dict

    @staticmethod
    def parse_entity(iter):
        """_summary_

        Args:
            iter (pd.Series): column families

        Returns:
            UserEntity: class instance
        """
        user_id = int(iter["user_id"])
        return UserEntity(
            user_idx=0,
            user_id=user_id,
        )


# end : class
