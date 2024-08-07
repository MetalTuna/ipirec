import math

from core import BaseDataSet, ItemEntity, UserEntity
from .base_distance_model import BaseDistanceModel


class Pearson(BaseDistanceModel):
    """
    - 요약:
        - Item-based correlations만 구현됐습니다.
        - 필요에 따라 enum으로 user / item을 구분하도록 구현하세요.
    """

    def __init__(
        self,
        dataset: BaseDataSet,
    ) -> None:
        '''
        self.id_to_mean_freq_dict = dict()
        """
        - 요약:
            - 평균 발생빈도 수
        - 구성:
            - Key: id (int)
            - Value: mean_freq (float)
        """
        '''
        super().__init__(dataset)

    def _preprocess_(self) -> None:
        """mean freq. 계산"""
        numer = 0.0
        denom = self.users_count

        for item_id in self.item_dict.keys():
            item: ItemEntity = self.item_dict[item_id]
            if not item_id in self.item_id_to_idx:
                continue
            item_idx: int = self.item_id_to_idx[item_id]
            numer = len(item.dict_of_users_decision["view"])
            # numer = numer / denom
            self.idx_to_mean_freq_array[item_idx] = numer / denom
            # self.id_to_mean_freq_dict.update({item_id: numer})
        # end : for (items)

    # end : protected void preprocess()

    def __distance__(self, item_x: int, item_y: int) -> None:
        x_idx: int = self.item_id_to_idx[item_x]
        y_idx: int = self.item_id_to_idx[item_y]
        x_item: ItemEntity = self.item_dict[item_x]
        y_item: ItemEntity = self.item_dict[item_y]
        denom_x = denom_y = numer = score_x = score_y = 0.0

        decision_users: set = x_item.dict_of_users_decision["view"].union(
            y_item.dict_of_users_decision["view"]
        )
        for user_id in decision_users:
            user: UserEntity = self.user_dict[user_id]
            _items: set = user.dict_of_decision_item_ids["view"]

            score_x = 1 if x_item.item_id in _items else 0
            score_x = score_x - self.idx_to_mean_freq_array[x_idx]
            score_y = 1 if y_item.item_id in _items else 0
            score_y = score_y - self.idx_to_mean_freq_array[y_idx]

            numer += score_x * score_y
            denom_x += math.pow(score_x, 2.0)
            denom_y += math.pow(score_y, 2.0)
        # end : for (users)

        denom_x = math.sqrt(denom_x) * math.sqrt(denom_y)
        numer = 0.0 if denom_x == 0.0 else numer / denom_x
        self.arr_similarties[x_idx][y_idx] = self.arr_similarties[y_idx][x_idx] = numer

    # end : private override void distance()

    def _postprocess_(self) -> None:
        pass

    # end : protected void postprocess()

    def _set_model_params_(
        self,
        model_params: dict,
    ) -> None:
        pass

    def create_models_parameters(self) -> dict:
        return dict()


# end : class
