"""
[작성일] 24.07.02. - 항목수를 줄이기 위한 모듈
[수정일] 
- YY.MM.DD. HH:MM. Description.
"""

# Build-in
import os
import sys

# 3rd Pty.
import pickle
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

UNDEF_IDS = -1
__FILE_DIR_PATH = os.path.dirname(__file__)
WORKSPACE_HOME = __FILE_DIR_PATH.replace(f"/{os.path.basename(__FILE_DIR_PATH)}", "")
# sys.path.append(WORKSPACE_HOME)
WORKSPACE_HOME = WORKSPACE_HOME.replace("/refine", "")
DATASET_DIR_HOME = f"{WORKSPACE_HOME}/data/colley"
print(WORKSPACE_HOME)
sys.path.append(WORKSPACE_HOME)

from core import (
    BaseAction,
    Machine,
    BaseRepository,
    UserEntity,
    TagEntity,
    DecisionType,
    DirectoryPathValidator,
)

# Custom LIB.
from .base_id_mapping import BaseIDMapping
from .board_id_mapping import BoardIDMapping
from .items_mapping import ItemsMapping
from .product_id_mapping import ProductIDMapping
from .colley_dates_queries import ColleyDatesQueries
from .colley_dates_repository import ColleyDatesRepository


class ColleyDatesItemsReductionRepository(ColleyDatesRepository):
    """
    - 요약:
        - 통합항목 집합의 크기를 줄일 목적으로, 항목에 긍정한 의사결정 수로 여과하는 클래스입니다.
    """

    def __init__(
        self,
        raw_data_path: str,
        db_src: Machine = Machine.E_MAC,
        begin_date_str: str = "2023-07-01",
        emit_date_str: str = "2023-12-31",
    ) -> None:
        super().__init__(
            raw_data_path,
            db_src,
            begin_date_str,
            emit_date_str,
        )
        self._item_id_to_decisions_freq_dict: dict = None
        """
        Key: item_id (int)
        Value: freq_dict (dict) {
            "view": 0,
            "like": 0,
            "purchase": 0,
            "all": 0,
            "positive": 0,
        }
        """

    # end : init()

    def items_reduction(
        self,
        positive_threshold: int = 10,
    ) -> None:
        self.load_data()
        self.convert_decision()

        self.__decisions_aggregation__()
        self.__positive_freq_reduction__(positive_threshold)
        self._remapping_()

        self.dump_data()

    # end : public void items_reduction()

    def __decisions_aggregation__(self) -> None:
        """항목에 대한 의사결정 수를 집계합니다."""
        print("> decisions_aggregation()")
        item_id_to_decisions_freq_dict = dict()
        for kwd in self.decisions_dict.keys():
            decisions_list: list = self.decisions_dict[kwd]
            for action in decisions_list:
                action: BaseAction
                if not action.item_id in item_id_to_decisions_freq_dict:
                    item_id_to_decisions_freq_dict.update(
                        {
                            action.item_id: {
                                "view": 0,
                                "like": 0,
                                "purchase": 0,
                                "all": 0,
                                "positive": 0,
                            }
                        }
                    )
                _freq_dict: dict = item_id_to_decisions_freq_dict[action.item_id]
                _freq_dict[kwd] += 1
                _freq_dict["positive"] += 0 if kwd == "view" else 1
                _freq_dict["all"] += 1
            # end : for (decisions)
        # end : for (decision_types)

        self._item_id_to_decisions_freq_dict = item_id_to_decisions_freq_dict
        self.__dump_decisions_freq__()

    # end : private void decisions_aggregation()

    def __positive_freq_reduction__(
        self,
        threshold: int,
    ) -> None:
        """
        - 요약:
            - 긍정한 의사결정 수가 임계 값보다 작으면 제거됩니다.

        - 매개변수:
            threshold (int): 긍정의 의사결정 수(좋아요 수 + 구매 수)
        """
        item_ids = list(self._item_id_to_decisions_freq_dict.keys())
        for item_id in item_ids:
            _freq_dict: dict = self._item_id_to_decisions_freq_dict[item_id]
            _freq: int = _freq_dict["positive"]
            if _freq >= threshold:
                continue
            item: ItemsMapping = self.item_dict.pop(item_id)

            for product_id in item.product_ids_set:
                if product_id in self.product_dict:
                    self.product_dict.pop(product_id)
            # end : for (products)

            for board_id in item.board_ids_set:
                if board_id in self.board_dict:
                    self.board_dict.pop(board_id)
            # end : for (boards)
        # end : for (items)

    # end : private void positive_freq_reduction()

    def _remapping_(self) -> None:
        """변경된 항목집합에 대한 관계를 재구성 합니다."""
        user_ids = set()

        # 삭제된 항목들이 의사결정 내역에 존재하지 않도록 제거합니다.
        for kwd in self.decisions_dict.keys():
            _decisions = list()
            for action in self.decisions_dict[kwd]:
                action: BaseAction
                if not action.item_id in self.item_dict:
                    continue
                _decisions.append(action)
                user_ids.add(action.user_id)
            # end : for (decisions)
            self.decisions_dict[kwd] = _decisions
        # end : for (decision_types)

        # 의사결정 내역이 없는 사용자 제거
        _idx = 0
        _user_dict = dict()
        for user_id in user_ids:
            user: UserEntity = self.user_dict[user_id]
            user.idx = _idx
            _user_dict.update({user_id: user})
            _idx += 1
        # end : for (users)
        self.user_dict = _user_dict

    # end : protected void remapping()

    def __dump_decisions_freq__(self) -> None:
        """각 항목들에 대한 의사결정 빈도 수를 출력합니다."""
        _file_path = str.format(
            "{0}/decisions_freq.csv",
            self._TEMP_DIR_PATH,
        )
        _kwds = ["view", "like", "purchase", "all", "positive"]
        with open(_file_path, "wt") as fout:
            # header
            __line = ""
            for _kwd in _kwds:
                __line += _kwd + ","
            __line = __line[0 : len(__line) - 1]
            fout.write(f"item_id,{__line}\n")

            # records
            for item_id in self._item_id_to_decisions_freq_dict.keys():
                _freq_dict: dict = self._item_id_to_decisions_freq_dict[item_id]
                __line = ""
                for _kwd in _kwds:
                    __line += f"{_freq_dict[_kwd]},"
                # end : for (kwds)
                __line = __line[0 : len(__line) - 1]
                fout.write(f"{item_id},{__line}\n")
            # end : for (items)
            fout.close()
        # end : StreamWriter()

    # end : private void dump_decisions_freq()


# end : class
