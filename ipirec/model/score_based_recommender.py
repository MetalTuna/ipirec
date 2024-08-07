from tqdm import tqdm

from core import BaseTrain, BaseRecommender, BaseAction, UserEntity


class ScoreBasedRecommender(BaseRecommender):
    """
    - 요약:
        - 의사결정 추정 점수를 기반으로 추천목록을 구성하는 추천기입니다.
    """

    def __init__(
        self,
        estimator: BaseTrain,
    ) -> None:
        super().__init__(estimator)

    # end : init()

    def _preprocess_(self) -> None:
        pass

    def prediction(self) -> None:
        """모든 항목에 대한 기대점수 구하기"""
        if self._is_predicted:
            return
        # 공간복잡도가 부족해서 사용자별 추천목록의 길이를 200으로 구성한다.
        # for user_id in self.user_dict.keys():
        for user_id in tqdm(
            iterable=self.user_dict.keys(),
            desc="Recommender.prediction()",
            total=self.users_count,
        ):
            user: UserEntity = self.user_dict[user_id]
            user.estimated_items_score_list.clear()
            for item_id in self.item_dict.keys():
                inst = BaseAction(user_id, item_id)
                inst.estimated_score = self.predict(user_id, item_id)
                user.estimated_items_score_list.append(inst)
            # end: for (items)
            user.estimated_items_score_list = sorted(
                user.estimated_items_score_list,
                key=lambda x: x.estimated_score,
                reverse=True,
            )[:200]
        # end : for (users)
        self._is_predicted = True

    # end : public void prediction()


# end : class
