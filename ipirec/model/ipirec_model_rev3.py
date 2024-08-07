"""
[작성일] 24.07.09 17:10. 개정안 초안작업 (구현 중)
[수정일]
- 24.07.10 10:00. 주석작성
"""

import numpy as np
from tqdm import tqdm

from core import BaseDataSet, TagEntity, UserEntity, ItemEntity

from colley import ColleyDataSetRev

# from .ipirec_model_rev import IPIRecModelRev
from .correlation_model import CorrelationModel


# class IPIRecModelRev3(IPIRecModelRev):
class IPIRecModelRev3(CorrelationModel):
    def __init__(self, dataset: BaseDataSet) -> None:
        CorrelationModel.__init__(self, dataset, dict())

    # end : init()

    def __tags_score__(self) -> None:
        _tags_len = self.tags_count
        tags_collection = list(self.tags_dict.keys())
        self.arr_tags_score = np.zeros(shape=(self.tags_count, self.tags_count))
        self.co_occur_ratio = np.zeros(shape=(self.tags_count, self.tags_count))

        for i in tqdm(
            iterable=range(_tags_len),
            desc="tags_score",
            total=_tags_len,
        ):
            x_name = tags_collection[i]
            x_inst: TagEntity = self.tags_dict[x_name]
            x_idx = self.tag_name_to_idx[x_name]
            for j in range(_tags_len):
                if i == j:
                    continue
                y_name = tags_collection[j]
                y_idx = self.tag_name_to_idx[y_name]
                y_inst: TagEntity = self.tags_dict[y_name]
                _u_item_ids: set = x_inst.item_ids_set.union(y_inst.item_ids_set)
                _n_item_ids: set = x_inst.item_ids_set.difference(y_inst.item_ids_set)
                _numer = len(_n_item_ids)
                _denom = len(_u_item_ids)
                _J = 0.0 if _denom == 0 else _numer / _denom
                self.arr_tags_score[x_idx][y_idx] = _J * (
                    self._arr_tags_bin_cos_items[x_idx][y_idx]
                    * self._arr_tags_bin_cos_users[x_idx][y_idx]
                    * self._arr_tags_freq_cos_items[x_idx][y_idx]
                    * self._arr_tags_freq_cos_users[x_idx][y_idx]
                )
            # end : for (dest_tags)
        # end : for (src_tags)

    # end : private override void tags_score()

    def __top_n_decision_tags__(self) -> None:
        if isinstance(self._dataset, ColleyDataSetRev):
            for user_id in self.user_dict.keys():
                user: UserEntity = self.user_dict[user_id]
                user.tags_decision_freq_dict.clear()
                for item_id in user.dict_of_decision_item_ids["view"]:
                    item: ItemEntity = self.item_dict[item_id]
                    for tag_name in item.tags_set:
                        if not tag_name in user.tags_decision_freq_dict:
                            user.tags_decision_freq_dict.update({tag_name: 0})
                        user.tags_decision_freq_dict[tag_name] += 1
                    # end : for (T(i))
                # end : for (I(u))

                user.top_n_decision_tags_set = user.dict_of_interaction_tags["all"]
            # end : for (users)
        else:
            super().__top_n_decision_tags__()

    # end : private override void top_n_decision_tags()

    @staticmethod
    def create_models_parameters() -> dict:
        """사용안함"""
        return dict()

    def _set_model_params_(self, model_params: dict) -> None:
        """사용안함"""
        return


# end : class
