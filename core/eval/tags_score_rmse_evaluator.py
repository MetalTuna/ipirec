"""
[작성일] 24.06.18 18:00. 
- 관련사항은 아래의 링크에서, [할 일 >> 실험과 결과관찰 >> 태그점수 적합셩 계산]을 참고
    - https://www.notion.so/colleykr/24-06-20-19beee73670f491f9533bcf361f985d0?pvs=4
"""

from pandas import DataFrame

# from core.model.base_recommender import BaseRecommender
from ..entity.base_action import BaseAction
from ..entity.user_entity import UserEntity
from ..model.base_recommender import BaseRecommender
from .base_evaluator import BaseEvaluator


class TagsScoreRMSEEvaluator(
    BaseEvaluator,
):

    def __init__(
        self,
        recommender: BaseRecommender,
        file_path: str,
    ) -> None:
        self.ACTUAL_SCORE = 1.0
        self.rmse_hits = 0.0
        self.no_of_hits = 0.0
        self.rmse_forall = 0.0
        super().__init__(recommender, file_path)

    # end : init()

    def _forall_rmse_(self) -> float:
        # RMSE (\forall Y)
        numer = denom = 0.0

        for inst in self.TEST_SET_LIST:
            inst: BaseAction
            user_id: int = inst.user_id
            item_id: int = inst.item_id

            user: UserEntity = self.user_dict[user_id]
            estimated_score = user.recommended_items_dict.get(
                item_id,
                self._recommender.predict(user_id, item_id),
            )
            numer += (self.ACTUAL_SCORE - estimated_score) ** 2.0
            denom += 1
        # end : for (test_set)
        numer = (numer / denom) ** 0.5

        return float(numer)

    # end : protected float global_score_distance()

    def _hits_rmse_(self) -> float:
        # RMSE (Y \cap \hat{Y})
        numer = denom = 0.0

        for inst in self.TEST_SET_LIST:
            inst: BaseAction
            user_id: int = inst.user_id
            item_id: int = inst.item_id

            user: UserEntity = self.user_dict[user_id]
            if not item_id in user.recommended_items_dict:
                continue
            estimated_score: float = user.recommended_items_dict[item_id]
            numer += (self.ACTUAL_SCORE - estimated_score) ** 2.0
            denom += 1
        # end : for (test_set)
        self.no_of_hits = denom
        numer = (numer / denom) ** 0.5

        return float(numer)

    # end : protected float personalized_score_distance()

    def eval(self) -> None:
        self.rmse_hits = self._hits_rmse_()
        self.rmse_forall = self._forall_rmse_()
        print(f"Hits RMSE: {self.rmse_hits}")
        print(f"ForAll RMSE: {self.rmse_forall}")

    # end : public void eval()

    def evlautions_summary_df(self) -> DataFrame:
        super().evlautions_summary_df()
        raise NotImplementedError()

    def __member_var_init__(self) -> None:
        super().__member_var_init__()
        raise NotImplementedError()


# end : class
