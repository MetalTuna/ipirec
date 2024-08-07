import os
from abc import *

from tqdm import tqdm
import pandas as pd

from ..defines import DecisionType


class BaseAction:
    """
    - 요약:
        - 항목에 대한 사용자의 의사결정 객체입니다.
    """

    def __init__(
        self,
        user_id: int = -1,
        item_id: int = -1,
    ):
        self.user_id = user_id
        self.item_id = item_id
        self.estimated_score = float()
        self.created_time = str()
        self.timestamp = int()
        """의사결정한 시간(unix_time)"""
        self._decision_type: DecisionType = None
        """의사결정에 대한 열거자"""

    @staticmethod
    def load_collection(file_path: str) -> list:
        print(f"[IO] {file_path}")
        actions_list = list()
        if not os.path.exists(file_path):
            raise FileNotFoundError()
        if not os.path.isfile(file_path):
            raise IsADirectoryError()

        dataset_name = os.path.dirname(file_path).split("/")[-1]
        dtype_str = os.path.basename(file_path).split("_")[-2]
        decision_type = DecisionType.from_str(dtype_str)
        log_kwd = "timestamp"

        ## 의사결정 내역에 중복이 있어서, 중복제거를 위한 작업이 추가됨. (24.05.31)
        # [Legacy]
        """
        if dataset_name == "ml":
            # movielens
            df = pd.read_csv(file_path)
            log_kwd = log_kwd if log_kwd in df.keys() else "created_time"
            for _, r in df.iterrows():
                inst = BaseAction(
                    user_id=int(r["user_id"]),
                    item_id=int(r["item_id"]),
                )
                inst.timestamp = int(r[log_kwd])
                inst._decision_type = decision_type
                actions_list.append(inst)
            # end : for (actions)
        elif dataset_name == "colley":
            # colley
            desc_str = "[LOAD] " + file_path.replace(
                os.path.dirname(file_path) + "/", ""
            )
            decision_df = pd.read_csv(file_path)

            for _, r in tqdm(
                iterable=decision_df.iterrows(),
                desc=desc_str,
                total=decision_df.shape[0],
            ):
                inst = BaseAction(
                    user_id=int(r["user_id"]),
                    item_id=int(r["item_id"]),
                )
                inst._decision_type = decision_type
                inst.created_time = r["created_time"]
                actions_list.append(inst)
            # end : for (actions)
        else:
            raise NotImplementedError()
        return actions_list
        """

        # [Replaced]
        decision_str_set = set()
        if dataset_name == "ml":
            # movielens
            df = pd.read_csv(file_path)
            log_kwd = log_kwd if log_kwd in df.keys() else "created_time"
            for _, r in df.iterrows():
                inst = BaseAction(
                    user_id=int(r["user_id"]),
                    item_id=int(r["item_id"]),
                )
                inst.timestamp = int(r[log_kwd])
                inst._decision_type = decision_type
                decision_str_set.add(inst.to_str())
                # actions_list.append(inst)
            # end : for (actions)
        elif dataset_name == "colley":
            # colley
            desc_str = "[LOAD] " + file_path.replace(
                os.path.dirname(file_path) + "/", ""
            )
            decision_df = pd.read_csv(file_path)

            for _, r in tqdm(
                iterable=decision_df.iterrows(),
                desc=desc_str,
                total=decision_df.shape[0],
            ):
                inst = BaseAction(
                    user_id=int(r["user_id"]),
                    item_id=int(r["item_id"]),
                )
                inst._decision_type = decision_type
                inst.created_time = r["created_time"]
                decision_str_set.add(inst.to_str())
                # actions_list.append(inst)
            # end : for (actions)
        else:
            raise NotImplementedError()

        for line in decision_str_set:
            inst = BaseAction.from_str(line)
            actions_list.append(inst)
        # end : for (decisions)

        return actions_list

    # end : public static list load_collection()

    @staticmethod
    def from_str(element_str: str):
        args = [v.strip() for v in element_str.split(",")]
        inst = BaseAction(
            user_id=int(args[0]),
            item_id=int(args[1]),
        )
        inst.created_time = args[2]
        return inst

    def to_str(self) -> str:
        return str.format(
            "{0},{1},{2}",
            self.user_id,
            self.item_id,
            self.created_time,
        )


# end : class
