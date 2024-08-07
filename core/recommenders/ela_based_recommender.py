"""
[작성일] 기억안남...
[수정일] 
- 2024.06.20 16:32., 실험모듈과 통합 (기존 모듈은 미완성 상태였음)
"""

# from core import BaseRecommender, UserEntity, ItemEntity, BaseTrain, DecisionType
from ..entity import UserEntity, ItemEntity
from ..defines import DecisionType
from ..model import BaseTrain, BaseRecommender


class ELABasedRecommender(BaseRecommender):
    """
    - 요약:
        - ELA를 조금 흉내낸 것: !ELA
        - 항목을 선호한 사용자들이 선호했던 항목들로 추천후보 목록을 구성하는 것
            - 공통 선호항목들로 교차추천하도록 구성하는 단순기능임
    """

    def __init__(
        self,
        estimator: BaseTrain,
    ) -> None:
        super().__init__(estimator)

    def _preprocess_(self):
        print(f"{type(self).__name__}.preprocess()")
        self.__candidate_items__()

    def __candidate_items__(self):
        """ELA의 게시글을 탐험방법으로 각 사용자별 항목의 추천 후보 목록을 만든다."""
        print(f"{type(self).__name__}.candidate_items()")
        if self._is_predicted:
            return

        ### 상품순회
        # 목표항목에 긍정한 사용자들이 긍정한 항목들의 집합을,
        # 목표항목을 열람한 사용자들의 추천 후보목록에 추가한다.
        for item_id in self.item_dict.keys():
            item: ItemEntity = self.item_dict[item_id]

            likes_user_ids: set = item.dict_of_users_decision[
                DecisionType.to_str(DecisionType.E_LIKE)
            ]
            purchases_user_ids: set = item.dict_of_users_decision[
                DecisionType.to_str(DecisionType.E_PURCHASE)
            ]

            positive_user_ids: set = likes_user_ids.union(purchases_user_ids)
            candidate_item_ids = set()
            ## positive items
            for user_id in positive_user_ids:
                user: UserEntity = self.user_dict[user_id]

                candidate_item_ids.update(
                    user.dict_of_decision_item_ids[
                        DecisionType.to_str(DecisionType.E_LIKE)
                    ]
                )
                candidate_item_ids.update(
                    user.dict_of_decision_item_ids[
                        DecisionType.to_str(DecisionType.E_PURCHASE)
                    ]
                )
            # end : for (positive_users)

            ## append_candidate_items
            for user_id in positive_user_ids:
                user: UserEntity = self.user_dict[user_id]
                user.candidate_item_ids_set.update(candidate_item_ids)
            # end : for (positive_users)
        # end : for (items)
        self._is_predicted = True

    # end : private void candidate_items()


# end : class
