"""
[수정일] 
- 24.06.26 17:24. 태그관계 계산이 재정의된 모듈입니다.
"""

from abc import *
import math
import os

import pickle
import numpy as np
from tqdm import tqdm

from core import *

NOT_INDEXED = -1
"""미 정의된 tag_index에 대한 정의"""


class CorrelationModel(BaseModel):
    """
    - 요약:
        - 태그 간의 점수를 구합니다.
    - 태그점수:
        - 태그 관계를 코사인으로 구합니다.
    """

    def __init__(
        self,
        dataset: BaseDataSet,
        model_params: dict,
    ) -> None:
        super().__init__(
            dataset=dataset,
            model_params=model_params,
        )
        self.arr_tags_score = None
        """태그 상관관계 점수 S[x,y]"""

        self.co_occur_ratio: np.ndarray = None
        self._arr_tags_bin_cos_users: np.ndarray = None
        self._arr_tags_freq_cos_users: np.ndarray = None
        self._arr_tags_bin_cos_items: np.ndarray = None
        self._arr_tags_freq_cos_items: np.ndarray = None

    # end : init()

    # models parameters
    def _set_model_params_(
        self,
        model_params: dict,
    ) -> None:
        kwd = "top_n_tags"
        if not kwd in model_params:
            raise KeyError()
        top_n_tags: int = model_params[kwd]
        self.TOP_N_TAGS = (
            1000 if top_n_tags > 1000 else (10 if top_n_tags < 0 else top_n_tags)
        )
        kwd = "co_occur_items_threshold"
        if not kwd in model_params:
            raise KeyError()
        self.CO_OCCUR_ITEMS_THRESHOLD: int = model_params[kwd]

    # end : pretected void override set_model_params()

    @staticmethod
    def create_models_parameters(
        top_n_tags: int,
        co_occur_items_threshold: int,
    ) -> dict:
        return {
            "top_n_tags": top_n_tags,
            "co_occur_items_threshold": co_occur_items_threshold,
        }

    # end : public static override dict create_models_parameters()

    def _preprocess_(self) -> None:
        ### 태그점수 계산을 위한 준비단계
        print(f"{type(self).__name__}.preprocess()")
        self.__top_n_decision_tags__()

    # end : protected override void preprocess()

    def _process_(self) -> None:
        self.__tags_sim_from_users__()
        self.__tags_sim_from_items__()
        self.__tags_score__()

    # end : protected override void process()

    def _postprocess_(self) -> None:
        pass

    # end : protected override void postprocess()

    def __tags_sim_from_items__(self) -> None:
        bin_cos = np.zeros(shape=(self.tags_count, self.tags_count))
        freq_cos = np.zeros(shape=(self.tags_count, self.tags_count))
        tags_list = list(self.tags_dict.keys())
        tag_name_to_tau_scores_dict = dict()

        for tag_name, inst in self.tags_dict.items():
            inst: TagEntity
            _freq = 0.0
            for item_id in inst.item_ids_set:
                item: ItemEntity = self.item_dict[item_id]
                _freq += len(item.tags_set)
            # end : for (items)
            _cnt = len(inst.item_ids_set)
            _freq = 0.0 if _cnt == 0.0 else _freq / _cnt
            tag_name_to_tau_scores_dict.update({tag_name: _freq})
        # end : for (tags)

        # inner product
        for i in range(self.tags_count):
            x_name: str = tags_list[i]
            x = self.tag_name_to_idx.get(x_name, NOT_INDEXED)
            if x == NOT_INDEXED:
                continue
            bin_cos[x][x] = freq_cos[x][x] = 1.0
            x_inst: TagEntity = self.tags_dict[x_name]
            for j in range(x + 1, self.tags_count):
                y_name: str = tags_list[j]
                y = self.tag_name_to_idx.get(y_name, NOT_INDEXED)
                if y == NOT_INDEXED:
                    continue
                y_inst: TagEntity = self.tags_dict[y_name]

                item_ids = set.union(x_inst.item_ids_set, y_inst.item_ids_set)
                denom_freq_x = denom_freq_y = denom_bin_x = denom_bin_y = numer_freq = (
                    numer_bin
                ) = _x = _y = 0.0

                for item_id in item_ids:
                    item: ItemEntity = self.item_dict[item_id]
                    _xb = 1 if x_name in item.tags_set else 0
                    _yb = 1 if y_name in item.tags_set else 0
                    _x: float = tag_name_to_tau_scores_dict.get(x_name, 0.0)
                    _y: float = tag_name_to_tau_scores_dict.get(y_name, 0.0)

                    denom_bin_x += _xb
                    denom_bin_y += _yb
                    numer_bin += _xb * _yb
                    denom_freq_x += _x**2
                    denom_freq_y += _y**2
                    numer_freq += _x * _y
                # end : for (occur_item_ids)

                denom_bin_x = (denom_bin_x**0.5) * (denom_bin_y**0.5)
                numer_bin = 0.0 if denom_bin_x == 0.0 else numer_bin / denom_bin_x
                bin_cos[x][y] = bin_cos[y][x] = numer_bin

                denom_freq_x = (denom_freq_x**0.5) * (denom_freq_y**0.5)
                numer_freq = 0.0 if denom_freq_x == 0.0 else numer_freq / denom_freq_x
                freq_cos[x][y] = freq_cos[y][x] = numer_freq
            # end : for (tags)
        # end : for (tags)

        self._arr_tags_bin_cos_items = bin_cos
        self._arr_tags_freq_cos_items = freq_cos

    # end : private void tags_sim_from_items()

    def __tags_sim_from_users__(self) -> None:
        bin_cos = np.zeros(shape=(self.tags_count, self.tags_count))
        freq_cos = np.zeros(shape=(self.tags_count, self.tags_count))
        tags_list = list(self.tags_dict.keys())

        for i in range(self.tags_count):
            x_name: str = tags_list[i]
            x = self.tag_name_to_idx.get(x_name, NOT_INDEXED)
            bin_cos[x][x] = freq_cos[x][x] = 1.0
            if x == NOT_INDEXED:
                continue
            x_inst: TagEntity = self.tags_dict[x_name]
            for j in range(x + 1, self.tags_count):
                y_name: str = tags_list[j]
                y = self.tag_name_to_idx.get(y_name, NOT_INDEXED)
                if y == NOT_INDEXED:
                    continue
                y_inst: TagEntity = self.tags_dict[y_name]
                co_occur_users = set.union(x_inst.user_ids_set, y_inst.user_ids_set)
                denom_freq_x = denom_freq_y = denom_bin_x = denom_bin_y = numer_freq = (
                    numer_bin
                ) = _x = _y = 0.0

                for user_id in co_occur_users:
                    user: UserEntity = self.user_dict[user_id]
                    _x = user.tags_decision_freq_dict.get(x_inst.tag_name, 0)
                    _y = user.tags_decision_freq_dict.get(y_inst.tag_name, 0)
                    _bx = 0 if _x == 0 else 1
                    _by = 0 if _y == 0 else 1

                    numer_bin += _bx * _by
                    denom_bin_x += _bx
                    denom_bin_y += _by
                    numer_freq += _x * _y
                    denom_freq_x += _x**2
                    denom_freq_y += _y**2
                # end : for (co_occur_users)

                denom_bin_x = (denom_bin_x**0.5) * (denom_bin_y**0.5)
                numer_bin = 0.0 if denom_bin_x == 0.0 else numer_bin / denom_bin_x
                bin_cos[x][y] = bin_cos[y][x] = numer_bin

                denom_freq_x = (denom_freq_x**0.5) * (denom_freq_y**0.5)
                numer_freq = 0.0 if denom_freq_x == 0.0 else numer_freq / denom_freq_x
                freq_cos[x][y] = freq_cos[y][x] = numer_freq
            # end : for (tags)
        # end : for (tags)

        self._arr_tags_bin_cos_users = bin_cos
        self._arr_tags_freq_cos_users = freq_cos

    # end : private void tags_sim_from_users()

    def __tags_score__(self) -> None:
        print(f"{type(self).__name__}.tags_score()")
        """
        _file_path = self._dump_dir_path + "/tags_score.bin"
        if self.IS_DEBUG_MODE:
            if os.path.exists(_file_path):
                with open(file=_file_path, mode="rb") as fin:
                    self.arr_tags_score = pickle.load(fin)
                    fin.close()
                # end : StreamReader()
                return
        # contains binary dump
        """

        tags_collection = list(self.tags_dict.keys())
        self.arr_tags_score = np.zeros(shape=(self.tags_count, self.tags_count))
        self.co_occur_ratio = np.zeros(shape=(self.tags_count, self.tags_count))

        # co_occur items freq.
        # for x_idx in range(self.tags_count):
        for i in tqdm(
            iterable=range(self.tags_count),
            desc="co_occur items freq.",
            total=self.tags_count,
        ):
            x_name = tags_collection[i]
            x_idx = self.tag_name_to_idx.get(x_name, NOT_INDEXED)
            if x_idx == NOT_INDEXED:
                continue
            x_inst: TagEntity = self.tags_dict[x_name]

            for j in range(i + 1, self.tags_count):
                y_name = tags_collection[j]
                y_idx = self.tag_name_to_idx.get(y_name, NOT_INDEXED)
                if y_idx == NOT_INDEXED:
                    continue
                y_inst: TagEntity = self.tags_dict[y_name]
                co_occur_items_len = len(
                    x_inst.item_ids_set.intersection(y_inst.item_ids_set)
                )
                self.arr_tags_score[x_idx][y_idx] = co_occur_items_len
            # end : for (tags - x_idx)
        # end : for (tags)

        max_freq: int = self.arr_tags_score.max()

        # for x_idx in range(self.tags_count):
        for i in tqdm(
            iterable=range(self.tags_count),
            desc="tags_score",
            total=self.tags_count,
        ):
            x_name = tags_collection[i]
            x_idx = self.tag_name_to_idx.get(x_name, NOT_INDEXED)
            if x_idx == NOT_INDEXED:
                continue
            x_inst: TagEntity = self.tags_dict[x_name]
            self.arr_tags_score[x_idx][x_idx] = 1.0

            if len(x_inst.item_ids_set) == 0:
                continue

            for j in range(i + 1, self.tags_count, 1):
                y_name = tags_collection[j]
                y_idx = self.tag_name_to_idx.get(y_name, NOT_INDEXED)
                if y_idx == NOT_INDEXED:
                    continue
                y_inst: TagEntity = self.tags_dict[y_name]
                # Co-occurrence items
                co_occur_items_len = self.arr_tags_score[x_idx][y_idx]
                # Logarithmic ratio score
                # co_occur_score: float = math.log(x=co_occur_items_len, base=self.CO_OCCUR_ITEMS_THRESHOLD)
                co_occur_score = (
                    0.0
                    if co_occur_items_len == 0
                    else math.log(
                        co_occur_items_len,
                        max_freq,
                    )
                )
                # Asymmetric corr score X and Y
                _cnt = len(x_inst.item_ids_set)
                co_occur_ratio_x = 0.0 if _cnt == 0.0 else co_occur_items_len / _cnt
                _cnt = len(y_inst.item_ids_set)
                co_occur_ratio_y = 0.0 if _cnt == 0 else co_occur_items_len / _cnt
                self.co_occur_ratio[x_idx][y_idx] = co_occur_ratio_x
                self.co_occur_ratio[y_idx][x_idx] = co_occur_ratio_y

                _tags_dist = co_occur_score * (
                    self._arr_tags_bin_cos_items[x_idx][y_idx]
                    * self._arr_tags_bin_cos_users[x_idx][y_idx]
                    * self._arr_tags_freq_cos_items[x_idx][y_idx]
                    * self._arr_tags_freq_cos_users[x_idx][y_idx]
                )
                self.arr_tags_score[x_idx][y_idx] = _tags_dist * co_occur_ratio_x
                self.arr_tags_score[y_idx][x_idx] = _tags_dist * co_occur_ratio_y
            # end : for (tags Y)
        # end : for (tags X)
        """
        if self.IS_DEBUG_MODE:
            with open(
                file=_file_path,
                mode="wb",
            ) as fout:
                pickle.dump(
                    self.arr_tags_score,
                    fout,
                )
                fout.close()
            # end : StreamWriter()
        """

    # end : private void tags_score()

    def __top_n_decision_tags__(self) -> None:
        # 태그를 추가하는 부분에서 집계하면 되는 것 아님??
        # 효율 좀 떨어져도 역할분리 목적으로 ㄱㄱ
        print(f"{type(self).__name__}.top_n_decision_tags()")
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
            user.top_n_decision_tags_set = {
                k
                for k, _ in sorted(
                    user.tags_decision_freq_dict.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[: self.TOP_N_TAGS]
            }
        # end : for (users)

    # end : private void top_n_decision_tags()


# end : class
