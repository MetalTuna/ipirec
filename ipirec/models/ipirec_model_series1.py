from abc import *
import math
import numpy as np
from tqdm import tqdm

from core import *


class IPIRecModelSeries1(BaseModel):
    """
    - 요약:
        - 태그 간의 점수를 구합니다.

    - 계산관련 노션링크:
        - https://www.notion.so/colleykr/IP-be052922bb704e32924307abc18acb62?pvs=4
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

        self.tags_weight_global: list = None
        self.tags_weight_local: dict = None
        self.tags_mean_freq = dict()
        """
        Key: tag_name (str)
        Value: mean_freq_score (float) 
            = |U(t)|^{-1} \\sum_{u \\in U(t)} |I(u,t)| + |I(t)|^{-1} \\sum_{i \\in I(t)} |U(i,t)|
        """

        self.co_occur_ratio: np.ndarray = None
        self._ub_pcc_co_occur_score: np.ndarray = None
        self._ib_pcc_co_occur_score: np.ndarray = None

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
        print(f"{type(self).__name__}.preprocess()")
        self.__top_n_decision_tags__()
        self.__mean_freq_tags__()

    # end : protected override void preprocess()

    def _process_(self) -> None:
        self.__item_based_tags_corr__()
        self.__user_based_tags_corr__()
        self.__tags_score__()

    # end : protected override void process()

    def _postprocess_(self) -> None:
        """
        - 후처리 (미사용)
        ===
        """
        pass

    # end : protected override void postprocess()

    def __tags_score__(self) -> None:
        print(f"{type(self).__name__}.tags_score()")

        tags_collection = list(self.tags_dict.keys())
        self.arr_tags_score = np.zeros(shape=(self.tags_count, self.tags_count))
        self.co_occur_ratio = np.zeros(shape=(self.tags_count, self.tags_count))

        for x_idx in tqdm(
            iterable=range(self.tags_count),
            desc="co_occur items freq.",
            total=self.tags_count,
        ):
            x_name = tags_collection[x_idx]
            x_inst: TagEntity = self.tags_dict[x_name]

            for y_idx in range(x_idx + 1, self.tags_count):
                y_name = tags_collection[y_idx]
                y_inst: TagEntity = self.tags_dict[y_name]
                co_occur_items_len = len(
                    x_inst.item_ids_set.intersection(y_inst.item_ids_set)
                )
                self.arr_tags_score[x_idx][y_idx] = co_occur_items_len
            # end : for (tags - x_idx)
        # end : for (tags)

        max_freq: int = self.arr_tags_score.max()
        tag_score = 0.0

        # for x_idx in range(self.tags_count):
        for x_idx in tqdm(
            iterable=range(self.tags_count),
            desc="tags_score",
            total=self.tags_count,
        ):
            x_name = tags_collection[x_idx]
            x_inst: TagEntity = self.tags_dict[x_name]
            self.arr_tags_score[x_idx][x_idx] = 1.0

            if len(x_inst.item_ids_set) == 0:
                continue

            for y_idx in range(x_idx + 1, self.tags_count, 1):
                y_name = tags_collection[y_idx]
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
                co_occur_ratio_x: float = co_occur_items_len / len(x_inst.item_ids_set)
                co_occur_ratio_y: float = (
                    0.0
                    if len(y_inst.item_ids_set) == 0
                    else co_occur_items_len / len(y_inst.item_ids_set)
                )
                self.co_occur_ratio[x_idx][y_idx] = co_occur_ratio_x
                self.co_occur_ratio[y_idx][x_idx] = co_occur_ratio_y

                tag_score = (
                    co_occur_score
                    * co_occur_ratio_x
                    * (
                        (0.5 * self._ub_pcc_co_occur_score[x_idx][y_idx])
                        + (0.5 * self._ib_pcc_co_occur_score[x_idx][y_idx])
                    )
                )
                self.arr_tags_score[x_idx][y_idx] = (
                    0.0 if math.nan == tag_score else tag_score
                )
                tag_score = (
                    co_occur_score
                    * co_occur_ratio_y
                    * (
                        (0.5 * self._ub_pcc_co_occur_score[y_idx][x_idx])
                        + (0.5 * self._ib_pcc_co_occur_score[y_idx][x_idx])
                    )
                )
                self.arr_tags_score[y_idx][x_idx] = (
                    0.0 if math.nan == tag_score else tag_score
                )
            # end : for (tags Y)
        # end : for (tags X)

    # end : private void tags_score()

    def __user_based_tags_corr__(self) -> None:
        print(f"{type(self).__name__}.user_based_tags_corr()")
        self._ub_pcc_co_occur_score = np.zeros(
            shape=(
                self.tags_count,
                self.tags_count,
            )
        )

        # 변경 후
        denom_x, denom_y, numer = 0.0, 0.0, 0.0
        score_x, score_y = 0.0, 0.0
        cnt = 0
        tag_names = list(self.tags_dict.keys())
        # self.tags_mean_freq
        # for x_idx in range(0, self.tags_count, 1):
        for x_idx in tqdm(
            iterable=range(0, self.tags_count, 1),
            desc="Pearson corr [users]: ",
            total=self.tags_count,
        ):
            denom_x = denom_y = numer = 0.0
            x_name = tag_names[x_idx]
            x_inst: TagEntity = self.tags_dict[x_name]

            for y_idx in range(x_idx + 1, self.tags_count, 1):
                y_name = tag_names[y_idx]
                y_inst: TagEntity = self.tags_dict[y_name]
                co_decision_users = x_inst.user_ids_set.intersection(
                    y_inst.user_ids_set
                )
                cnt = len(co_decision_users)
                if cnt == 0:
                    continue
                for user_id in co_decision_users:
                    user: UserEntity = self.user_dict[user_id]
                    if not x_name in user.dict_of_interaction_tags["view"]:
                        continue
                    if not y_name in user.dict_of_interaction_tags["view"]:
                        continue
                    cnt = len(
                        x_inst.item_ids_set.intersection(
                            user.dict_of_decision_item_ids["view"]
                        )
                    )
                    if cnt == 0:
                        continue
                    score_x = cnt - self.tags_mean_freq[x_name]
                    cnt = len(
                        y_inst.item_ids_set.intersection(
                            user.dict_of_decision_item_ids["view"]
                        )
                    )
                    if cnt == 0:
                        continue
                    score_y = cnt - self.tags_mean_freq[y_name]

                    numer += score_x * score_y
                    denom_x += math.pow(score_x, 2.0)
                    denom_y += math.pow(score_y, 2.0)
                # end : for (users)

                denom_x = math.sqrt(denom_x) * math.sqrt(denom_y)
                numer = 0.0 if denom_x == 0.0 else numer / denom_x
                self._ub_pcc_co_occur_score[x_idx][y_idx] = self._ub_pcc_co_occur_score[
                    y_idx
                ][x_idx] = numer
            # end : for (tag_y)
        # end : for (tag_x)

    # end : private void user_based_tags_corr()

    def __item_based_tags_corr__(self) -> None:
        print(f"{type(self).__name__}.item_based_tags_corr()")
        self._ib_pcc_co_occur_score = np.zeros(
            shape=(
                self.tags_count,
                self.tags_count,
            )
        )
        denom_x, denom_y, numer = 0.0, 0.0, 0.0
        score_x, score_y = 0.0, 0.0
        cnt = 0
        tag_names = list(self.tags_dict.keys())

        for x_idx in tqdm(
            iterable=range(0, self.tags_count, 1),
            desc="Pearson corr [items]: ",
            total=self.tags_count,
        ):
            denom_x = denom_y = numer = 0.0
            x_name = tag_names[x_idx]
            x_inst: TagEntity = self.tags_dict[x_name]

            for y_idx in range(x_idx + 1, self.tags_count, 1):
                y_name = tag_names[y_idx]
                y_inst: TagEntity = self.tags_dict[y_name]

                co_occur_items = x_inst.item_ids_set.intersection(y_inst.item_ids_set)
                cnt = len(co_occur_items)
                if cnt == 0:
                    continue
                for item_id in co_occur_items:
                    item: ItemEntity = self.item_dict[item_id]
                    cnt = len(
                        x_inst.user_ids_set.intersection(item.set_of_view_user_ids)
                    )
                    if cnt == 0:
                        continue
                    score_x = cnt - self.tags_mean_freq[x_name]
                    cnt = len(
                        y_inst.user_ids_set.intersection(item.set_of_view_user_ids)
                    )
                    if cnt == 0:
                        continue
                    score_y = cnt - self.tags_mean_freq[y_name]

                    numer += score_x * score_y
                    denom_x += math.pow(score_x, 2.0)
                    denom_y += math.pow(score_y, 2.0)
                # end : for (items)

                denom_x = math.sqrt(denom_x) * math.sqrt(denom_y)
                numer = 0.0 if denom_x == 0.0 else numer / denom_x
                # symmetric corr
                self._ib_pcc_co_occur_score[x_idx][y_idx] = self._ib_pcc_co_occur_score[
                    y_idx
                ][x_idx] = numer
            # end : for (tag_y)
        # end : for (tag_x)

    # end : private void item_based_tags_corr()

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

    def __mean_freq_tags__(self) -> None:
        print(f"{type(self).__name__}.mean_freq_tags()")
        numer, denom, temp_score, cnt = 0.0, 0.0, 0.0, 0
        for tag_name in self.tags_dict.keys():
            numer = denom = 0.0
            tag: TagEntity = self.tags_dict[tag_name]
            for user_id in tag.user_ids_set:
                user: UserEntity = self.user_dict[user_id]
                if not tag_name in user.dict_of_interaction_tags["view"]:
                    continue
                cnt = len(
                    tag.item_ids_set.intersection(
                        user.dict_of_decision_item_ids["view"]
                    )
                )
                if cnt == 0:
                    continue
                numer += cnt
                denom += 1
            # end : for (users)
            temp_score = 0.0 if denom == 0.0 else numer / denom

            numer = denom = 0.0
            for item_id in tag.item_ids_set:
                item: ItemEntity = self.item_dict[item_id]
                cnt = len(item.dict_of_users_decision["view"])
                if cnt == 0:
                    continue
                numer += cnt
                denom += 1
            # end : for (items)
            numer = 0.0 if denom == 0.0 else numer / denom

            self.tags_mean_freq.update({tag_name: (temp_score + numer)})
        # end : for (tags)

    # end : private void mean_freq_tags()


# end : class
