"""
작성일: 24.05.14. 
TagsCorrelationEstimator 
"""

import numpy as np

from core import *
from .base_corr_estimator import BaseCorrelationEstimator


class BiasedCorrelationEstimator(BaseCorrelationEstimator):
    def __init__(
        self,
        model: BaseModel,
        model_params: dict,
    ) -> None:
        super().__init__(
            model,
            model_params,
        )
        self.__append_biases__()

    # end : init()

    def __append_biases__(self) -> None:
        """append tendencies axis"""

        print(f"{type(self).__name__}.append_biases()")
        """[view_list] mean freq."""
        decision_type = DecisionType.E_VIEW
        kwd = DecisionType.to_str(decision_type)
        decisions_dense_ratio: float = len(self.view_list) / (
            self.users_count * self.items_count
        )
        """\\mu (\\tau) = |R(Views)| / (|U| * |I|) """

        ### [Users]
        for user_id in self.user_dict.keys():
            user: UserEntity = self.user_dict[user_id]
            idx = self.user_id_to_idx[user_id]
            self._arr_users_bias[idx] = (
                len(user.dict_of_decision_item_ids[kwd]) / self.items_count
            ) - decisions_dense_ratio
        # end : for (users)

        ### [Items]
        for item_id in self.item_dict.keys():
            item: ItemEntity = self.item_dict[item_id]
            # is_filtered_item?
            if not item_id in self.item_id_to_idx:
                continue
            idx = self.item_id_to_idx[item_id]
            self._arr_items_bias[idx] = (
                len(item.dict_of_users_decision[kwd]) / self.users_count
            ) - decisions_dense_ratio
        # end : for (items)

        ### [Tags]
        denom = len(self.view_list)
        for tag_name in self.tags_dict.keys():
            idx = self.tag_name_to_idx[tag_name]
            inst: TagEntity = self.tags_dict[tag_name]
            self._arr_tags_bias[idx] = (
                inst.decisions_freq_dict[kwd] / denom
            ) - decisions_dense_ratio
        # end : for (tags)

        for x_name in self.tags_dict.keys():
            x_idx: int = self.tag_name_to_idx[x_name]
            x_inst: TagEntity = self.tags_dict[x_name]

            for y_name in self.tags_dict.keys():
                y_idx: int = self.tag_name_to_idx[y_name]
                if x_idx == y_idx:
                    continue

                y_inst: TagEntity = self.tags_dict[y_name]
                user_ids: set = x_inst.user_ids_set.union(y_inst.user_ids_set)
                users_axis = 0.0
                for user_id in user_ids:
                    uidx: int = self.user_id_to_idx[user_id]
                    users_axis += self._arr_users_bias[uidx]
                # end : for (users)
                users_axis = 0.0 if len(user_ids) == 0 else users_axis / len(user_ids)

                item_ids: set = x_inst.item_ids_set.union(y_inst.item_ids_set)
                items_axis = 0.0
                for item_id in item_ids:
                    if not item_id in self.item_id_to_idx:
                        continue
                    iidx: int = self.item_id_to_idx[item_id]
                    items_axis += self._arr_items_bias[iidx]
                # end : for (items)
                items_axis = 0.0 if len(item_ids) == 0 else items_axis / len(item_ids)
                axis_score = (
                    users_axis
                    + items_axis
                    + ((self._arr_tags_bias[x_idx] + self._arr_tags_bias[y_idx]) / 2.0)
                )
                ### 읽기전용으로 사용할 생각이었으니, setter 추가하지 말고 직접 참조하도록 구현
                # self.arr_tags_score[x_idx][y_idx] += axis_score
                self.model.arr_tags_score[x_idx][y_idx] += axis_score
            # end : for (tags)
        # end : for (tags)

    # end : private void append_biases()

    def __member_vars_allocation__(self) -> None:
        super().__member_vars_allocation__()

        ### member variables
        self._arr_users_bias = np.zeros(shape=(self.users_count, 1))
        self._arr_items_bias = np.zeros(shape=(self.items_count, 1))
        self._arr_tags_bias = np.zeros(shape=(self.tags_count, 1))
        self._DEFAULT_VOTING = float()
        """
        - 요약:
            - NT(u)에 target tag가 미포함일 때, 반환되는 값입니다.
            - 0~1사이의 값을 갖습니다. (절대 값)
        """

    # end : protected override void member_vars_allocation()


# end : class
