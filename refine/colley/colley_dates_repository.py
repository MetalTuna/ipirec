"""
[작성일] 24.07.01. - 실험용 데이터 셋(주어진 기간에 속하는 의사결정 내역들에 관한) 생성모듈
[수정일] 
- 24.07.02. 15:10. 기능구현 중
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


class ColleyDatesRepository(
    BaseRepository,
    ColleyDatesQueries,
):
    """
    - 요약:
        - 태그의 규모축소를 통한 데이터 셋을 생성하는 클래스입니다.
    """

    def __init__(
        self,
        raw_data_path: str,
        db_src: Machine = Machine.E_MAC,
        begin_date_str: str = "2023-07-01",
        emit_date_str: str = "2023-12-31",
    ) -> None:
        BaseRepository.__init__(
            self,
            raw_data_path,
            db_src,
        )
        ColleyDatesQueries.__init__(
            self,
            begin_date_str,
            emit_date_str,
        )
        self._TEMP_DIR_PATH = f"{self._raw_data_path}/temp"
        if not os.path.exists(self._TEMP_DIR_PATH):
            os.mkdir(self._TEMP_DIR_PATH)

        self._EXPORT_DIR_PATH = self._raw_data_path
        self.__LEMMA_DICT_DIR_PATH = f"{WORKSPACE_HOME}/resources/tags_dictionary"
        self.aliases_dict: dict = None
        """
        - 요약:
            - 태그들의 통합사전입니다. 
            - 모든 공백을 제거하고, 알파벳은 대문자로 변경해 구성됩니다.
            
        >>> `alias.csv`와 `synonym.csv`로 생성됩니다. 
        
        - 구성:
            - Key: to (str)
                
            - Value: from (str) - x,y,z, ...
        """
        self.alias_set: set = None
        """
        - 요약:
            - 변환된 태그들의 집합입니다.
        - 용도:
            - 이 변수를 참조해서 원시 태그들을 선별합니다.
                - 있으면 사용, 없으면 제거
        """

        self.user_dict: dict = None
        """
        Key: user_id (int)
        Value: inst (UserEntity)
        """

        self.board_dict: dict = None
        """
        Key: board_id (int)
        Value: inst (BoardIDMapping)
        """
        self.product_dict: dict = None
        """
        Key: product_id (int)
        Value: inst (ProductIDMapping)
        """
        self.item_dict: dict = None
        """
        Key: item_id (int)
        Value: inst (ItemsMapping)
        """
        self.tags_dict: dict = None
        """
        Key: tag_name (str)
        Value: inst (TagEntity)
        """
        self.decisions_dict: dict = None
        """
        Key: DecisionTypeKwd (str)
        Value: list (BaseAction)
        """
        self._dummy_list = list()

    # end : init()

    def load_data(self):
        print(f"[{type(self).__name__}] load_data()")
        ## preprocess
        self.__read_lemma_tokens__()
        self.__build_tags_dict__()
        self.__metadata_clone__()
        ## aggregation
        self.__decisions_clone__()

        ## mapping
        self.__build_item_ids__()
        self.__append_metadata__()

    # end : public override Any load_data()

    def convert_decision(self) -> None:
        """
        - 요약:
            - 통합항목을 재구성과 의사결정 내역에 속하는 사용자들을 선별하는 함수입니다.
        - 기능:
            - 태그가 없는 항목들을 제거
            - 제거된 항목에 속한 의사결정들을 제거
            - 의사결정 내역이 없는 사용자들을 제거
        """
        print(f"[{type(self).__name__}] convert_decision()")
        # 태그가 없는 항목들 제거
        self.__items_filtering__()

        # 원시 의사결정 내역을 만들고 통합항목으로 색인합니다.
        # 이 과정에서 의사결정 내역이 없는 사용자들을 제거한 후, 재색인합니다.
        self.__build_decisions_data__()

        # 의사결정 내역들의 통합으로 발생할 수 있는 중복된 의사결정들을 제거합니다.
        # (좋아요와 구매내역에 대한 의사결정 내역의 일부가 중복됐고, 이 작업으로 중복이 제거됐음)
        ## ex. 데이터 베이스에 중복된 레코드가 있거나,
        ## 통합항목으로 변환하면서 구매옵션만 상이할 경우에는 의사결정된 시간까지 동일하게 생성될 수 있습니다.
        self.__decisions_filtering__()

    # end : public void convert_decision()

    def dump_data(self):
        """
        - 실험용 데이타 가공과 출력
        """
        print(f"[{type(self).__name__}] dump_data()")
        _EXPORT_DIR_PATH = f"{self._raw_data_path}/refined"
        if not DirectoryPathValidator.exist_dir(_EXPORT_DIR_PATH):
            DirectoryPathValidator.mkdir(_EXPORT_DIR_PATH)

        ## [users]
        _idx = 0
        # users >> [user, user_interest_tag]_list.csv
        with open(
            file=f"{_EXPORT_DIR_PATH}/user_list.csv",
            mode="wt",
        ) as ul_fout, open(
            file=f"{_EXPORT_DIR_PATH}/user_interest_tag_list.csv",
            mode="wt",
        ) as ut_fout:
            # [HEADER]
            ul_fout.write(",user_id\r\n")
            ut_fout.write(",user_id,tag\r\n")
            _idx = 0
            for user_id in self.user_dict.keys():
                user: UserEntity = self.user_dict[user_id]
                ul_fout.write(f"{user.idx},{user.user_id}\r\n")
                for tag in user.set_of_interest_tags:
                    ut_fout.write(f"{_idx},{user.user_id},{tag}\r\n")
                    _idx += 1
                # end : for (interest_tags)
            # end : for (users)

            ut_fout.close()
            ul_fout.close()
        # end : StreamWriter()

        # relations_binary_dict >> [board_item, item_list, product_item].bin
        self.__dump_bins__(self.board_dict, "board_item", _EXPORT_DIR_PATH)
        self.__dump_bins__(self.item_dict, "item_list", _EXPORT_DIR_PATH)
        self.__dump_bins__(self.product_dict, "product_item", _EXPORT_DIR_PATH)

        # decisions_list >> [view, like, purchase]_list.csv
        for kwd in self.decisions_dict.keys():
            _file_path = f"{_EXPORT_DIR_PATH}/{kwd}_list.csv"
            _decision_list: list = self.decisions_dict[kwd]
            with open(file=_file_path, mode="wt") as fout:
                # [HEADER]
                fout.write("user_id,item_id,created_time\r\n")
                for inst in _decision_list:
                    inst: BaseAction
                    fout.write(f"{inst.to_str()}\r\n")
                # end : for (decisions)
                fout.close()
            # end : StreamWriter()
        # end : for (decision_types)

        # tags >> tag_list.csv
        _file_path = f"{_EXPORT_DIR_PATH}/tag_list.csv"
        with open(file=_file_path, mode="wt") as fout:
            fout.write(",tag_id,tag\r\n")
            _idx = 0
            for tag_name in self.tags_dict.keys():
                inst: TagEntity = self.tags_dict[tag_name]
                fout.write(f"{_idx},{inst.id},{tag_name}\r\n")
                _idx += 1
            # end : for (tags)
            fout.close()
        # end : StreamWriter()

        # description
        print(
            str.format(
                "[INFO]\n- {0:9s}: {1:9d}\n- {2:9s}: {3:9d}\n- {4:9s}: {5:9d}",
                "Users",
                len(self.user_dict),
                "Items",
                len(self.item_dict),
                "Tags",
                len(self.tags_dict),
            )
        )
        for kwd in self.decisions_dict.keys():
            print(
                str.format(
                    "- {0:9s}: {1:9d}",
                    kwd,
                    len(
                        self.decisions_dict[kwd],
                    ),
                )
            )
        # end : for (decision_types)

        self.__clean_up__()

    # end : public override void dump_data()

    def __clean_up__(self) -> None:
        self._dummy_list
        pass

    # private void clean_up()

    def __items_filtering__(self) -> None:
        """태그가 없는 항목들을 제거합니다."""
        print("> items_filtering()")
        item_ids = list(self.item_dict.keys())
        for item_id in item_ids:
            item: ItemsMapping = self.item_dict[item_id]
            if len(item.tags_set) != 0:
                continue

            for product_id in item.product_ids_set:
                product: ProductIDMapping = self.product_dict.get(product_id, UNDEF_IDS)
                if product != UNDEF_IDS:
                    self.product_dict.pop(product_id)
            # end : for (product_ids)

            for board_id in item.board_ids_set:
                board: BoardIDMapping = self.board_dict.get(board_id, UNDEF_IDS)
                if board != UNDEF_IDS:
                    self.board_dict.pop(board_id)
            # end : for (board_ids)
            self.item_dict.pop(item_id)
        # end : for (items)

        ## re-indexing
        item_id = 0
        _item_dict = dict()
        for _ in self.item_dict.keys():
            item: ItemsMapping = self.item_dict[_]
            _item_dict.update({item_id: item})
            for product_id in list(item.product_ids_set):
                product: ProductIDMapping = self.product_dict.get(product_id, UNDEF_IDS)
                if product == UNDEF_IDS:
                    item.product_ids_set.remove(product_id)
                else:
                    product.item_id = item_id
            # end : for (product_ids)
            for board_id in list(item.board_ids_set):
                board: BoardIDMapping = self.board_dict.get(board_id, UNDEF_IDS)
                if board == UNDEF_IDS:
                    item.board_ids_set.remove(board_id)
                else:
                    board.item_id = item_id
            # end : for (board_ids)
            item_id += 1
        # end : for (items)

        self.item_dict = _item_dict

    # end : private void items_filtering()

    def __build_decisions_data__(self) -> None:
        """여과된 통합항목으로 의사결정 내역을 재구성합니다."""
        print("> build_decisions_data()")
        user_ids = set()
        view_list = list()
        like_list = list()
        purchase_list = list()

        def __get_decisions_list__(__df: DataFrame, __kwd: str) -> list:
            __decisions = list()
            __ref_dict = self.product_dict if __kwd == "product_id" else self.board_dict
            for _, __r in __df.iterrows():
                __user_id = int(__r["user_id"])
                __target_id = int(__r[__kwd])
                __inst: BaseIDMapping = __ref_dict.get(__target_id, UNDEF_IDS)
                if __inst == UNDEF_IDS:
                    continue
                __item_id = __inst.item_id
                __action = BaseAction(__user_id, __item_id)
                __action.created_time = str(__r["created_time"]).strip()
                user_ids.add(__user_id)
                __decisions.append(__action)
            # end : for (decisions)
            return __decisions

        # [products]
        _kwd = "product_id"
        view_list.extend(__get_decisions_list__(self._product_views_df, _kwd))
        like_list.extend(__get_decisions_list__(self._product_likes_df, _kwd))
        purchase_list.extend(__get_decisions_list__(self._product_scrap_df, _kwd))
        purchase_list.extend(__get_decisions_list__(self._purchase_products_df, _kwd))

        # [boards]
        _kwd = "board_id"
        view_list.extend(__get_decisions_list__(self._board_views_df, _kwd))
        like_list.extend(__get_decisions_list__(self._board_likes_df, _kwd))
        purchase_list.extend(__get_decisions_list__(self._board_scrap_df, _kwd))
        purchase_list.extend(__get_decisions_list__(self._board_owners_df, _kwd))
        purchase_list.extend(__get_decisions_list__(self._board_request_df, _kwd))

        ## re-indexing
        _user_dict = dict()
        idx = 0
        for user_id in user_ids:
            user: UserEntity = self.user_dict[user_id]
            user._idx = idx
            _user_dict.update({user_id: user})
            idx += 1
        # end : for (user_ids)
        self.user_dict = _user_dict

        self.decisions_dict = {
            DecisionType.to_str(DecisionType.E_VIEW): view_list,
            DecisionType.to_str(DecisionType.E_LIKE): like_list,
            DecisionType.to_str(DecisionType.E_PURCHASE): purchase_list,
        }

    # end : private void build_decisions_data()

    def __decisions_filtering__(self) -> None:
        """중복된 의사결정들을 제거합니다."""
        print("> decisions_filtering()")
        ## inst -> str -> inst

        def __reduction__(__decisions: list) -> list:
            __decisions_str = set()
            __reduced_decisions = list()
            for __inst in __decisions:
                __inst: BaseAction
                __decisions_str.add(__inst.to_str())
            # end : for (decisions)
            for __line in __decisions_str:
                __reduced_decisions.append(BaseAction.from_str(__line))
            # end : for (reduced)
            return __reduced_decisions

        for _kwd in self.decisions_dict.keys():
            _decisions: list = self.decisions_dict[_kwd]
            _decisions = __reduction__(_decisions)
            self.decisions_dict[_kwd] = _decisions
        # end : for (decision_types)

    # end : private void decisions_filtering()

    def __read_lemma_tokens__(self) -> None:
        """통합사전 구성"""
        print("> read_lemma_tokens()")
        _aliases_dict = dict()
        _alias_tok_set = set()
        ## 통합목록 구성
        if not os.path.exists(self.__LEMMA_DICT_DIR_PATH):
            raise FileNotFoundError()

        ## alias.csv
        _file_path = f"{self.__LEMMA_DICT_DIR_PATH}/alias.csv"
        if not os.path.exists(_file_path):
            raise FileNotFoundError()
        _desc = "alias_dict"
        _df = pd.read_csv(_file_path)
        # for _, r in _df.iterrows():
        for _, r in tqdm(
            iterable=_df.iterrows(),
            desc=_desc,
            total=_df.shape[0],
        ):
            _to = self.__replace_tok__(r["to"])
            _aliases = {
                self.__replace_tok__(alias) for alias in str(r["from"]).split(",")
            }
            _alias_tok_set.add(_to)
            for _alias in _aliases:
                if _alias in _aliases_dict:
                    continue
                _aliases_dict.update({_alias: _to})
        # end : for (alises)

        ## synonym.csv
        _file_path = f"{self.__LEMMA_DICT_DIR_PATH}/synonym.csv"
        if not os.path.exists(_file_path):
            raise FileNotFoundError()
        _desc = "synonym_dict"
        _df = pd.read_csv(_file_path)
        # for _, r in _df.iterrows():
        for _, r in tqdm(
            iterable=_df.iterrows(),
            desc=_desc,
            total=_df.shape[0],
        ):
            _to = self.__replace_tok__(r["to"])
            _aliases = {
                self.__replace_tok__(alias) for alias in str(r["from"]).split(",")
            }
            _alias_tok_set.add(_to)
            for _alias in _aliases:
                if _alias in _aliases_dict:
                    continue
                _aliases_dict.update({_alias: _to})
        # end : for (alises)

        self.aliases_dict = _aliases_dict
        self.alias_set = _alias_tok_set

    # end : private void read_lemma_tokens()

    def __build_tags_dict__(self) -> None:
        print("> build_tags_dict()")
        _RESOURCE_DIR_PATH = f"{WORKSPACE_HOME}/resources/tags_info"
        tags_dict = dict()
        idx = 0

        # licenses_list
        _file_path = f"{_RESOURCE_DIR_PATH}/license_list.csv"
        for _, r in pd.read_csv(_file_path).iterrows():
            tag = self.__replace_tok__(r["name"])
            tag = self.aliases_dict.get(tag, tag)
            if tag in tags_dict:
                continue
            inst = TagEntity(idx, tag)
            tags_dict.update({tag: inst})
            idx += 1
        # end : for (licenses)

        # categories_list
        _file_path = f"{_RESOURCE_DIR_PATH}/category_list.csv"
        for _, r in pd.read_csv(_file_path).iterrows():
            tag = self.__replace_tok__(r["name"])
            tag = self.aliases_dict.get(tag, tag)
            if tag in tags_dict:
                continue
            inst = TagEntity(idx, tag)
            tags_dict.update({tag: inst})
            idx += 1
        # end : for (categories)

        self.tags_dict = tags_dict

    # end : private void build_tags_dict()

    def __metadata_clone__(self) -> None:
        """
        - 실험용 데이터 셋 구성에 필요한 자원들을 데이터베이스에 요청하고, 객체화합니다.
        """
        print("> metadata_clone()")
        # self._all_users_df = self.__contains_data__("all_users", self.SQL_ALL_USER_LIST)
        self._all_users_interest_tags_df = self.__contains_data__(
            "all_users_interest_tags", self.SQL_ALL_USER_INTEREST_TAG_LIST
        )

        self._all_boards_df = self.__contains_data__(
            "all_boards", self.SQL_ALL_BOARD_LIST
        )
        self._all_board_tags_df = self.__contains_data__(
            "all_board_tags", self.SQL_ALL_BOARD_TAG_LIST
        )

        self._all_products_df = self.__contains_data__(
            "all_products", self.SQL_ALL_PRODUCT_LIST
        )
        self._all_product_tags_df = self.__contains_data__(
            "all_product_tags", self.SQL_ALL_PRODUCT_TAG_LIST
        )

        self._all_board_product_df = self.__contains_data__(
            "all_board_product", self.SQL_ALL_BOARD_PRODUCT_LIST
        )

    # end : private void metadata_clone()

    def __decisions_clone__(self) -> None:
        """의사결정 내역들을 데이터 베이스에서 가져옵니다."""
        print("> decisions_clone()")

        # Purchases
        self._purchase_products_df = self.__contains_data__(
            "purchase_products", self.SQL_PURCHASE_LIST
        )
        self._product_scrap_df = self.__contains_data__(
            "product_scrap", self.SQL_PRODUCT_SCRAP_LIST
        )
        self._board_owners_df = self.__contains_data__(
            "board_owners", self.SQL_OWNED_USER_LIST
        )
        self._board_request_df = self.__contains_data__(
            "board_request", self.SQL_BOARD_REQUEST_LIST
        )
        self._board_scrap_df = self.__contains_data__(
            "board_scrap", self.SQL_BOARD_SCRAP_LIST
        )

        # Likes
        self._board_likes_df = self.__contains_data__(
            "board_likes", self.SQL_LIKE_BOARD_LIST
        )
        self._product_likes_df = self.__contains_data__(
            "product_likes", self.SQL_LIKE_PRODUCT_LIST
        )

        # Views
        self._board_views_df = self.__contains_data__(
            "board_views", self.SQL_OPEN_BOARD_LIST
        )
        self._product_views_df = self.__contains_data__(
            "product_views", self.SQL_OPEN_PRODUCT_LIST
        )

    # end : private void decisions_clone()

    def __build_item_ids__(self) -> None:
        """통합항목 생성: 의사결정에 참여한 사용자, 게시글, 상품들의 식별자를 색인합니다."""

        print("> build_item_ids()")
        item_id = 0
        user_dict = dict()
        board_dict = dict()
        product_dict = dict()
        item_dict = dict()

        self.user_dict = self.__load_bins__("user_dict")
        self.board_dict = self.__load_bins__("board_dict")
        self.product_dict = self.__load_bins__("product_dict")
        self.item_dict = self.__load_bins__("item_dict")
        if (
            (self.user_dict != None)
            and (self.board_dict != None)
            and (self.product_dict != None)
            and (self.item_dict != None)
        ):
            return
        else:
            pass

        def __append_user__(_r) -> int:
            __uid = int(_r["user_id"])
            if not __uid in user_dict:
                __inst = UserEntity(user_id=__uid)
                user_dict.update({__uid: __inst})
            return __uid

        def __append_board__(_r, _item_id: int) -> int:
            _bid = int(_r["board_id"])
            _inst: int = board_dict.get(_bid, UNDEF_IDS)
            if _inst == UNDEF_IDS:
                _inst = BoardIDMapping(_bid, _item_id, "")
                board_dict.update({_bid: _inst})
                _inst = ItemsMapping(_item_id, "")
                item_dict.update({_item_id: _inst})
                _item_id += 1
            _inst: BoardIDMapping = board_dict[_bid]
            _item: ItemsMapping = item_dict[_inst.item_id]
            _item.board_ids_set.add(_bid)
            return _item_id

        def __append_product__(_r, _item_id: int) -> int:
            _pid = int(r["product_id"])
            _inst: int = product_dict.get(_pid, UNDEF_IDS)
            if _inst == UNDEF_IDS:
                _inst = ProductIDMapping(_pid, _item_id, "")
                product_dict.update({_pid: _inst})
                _inst = ItemsMapping(_item_id, "")
                item_dict.update({_item_id: _inst})
                _item_id += 1
            _inst: ProductIDMapping = product_dict[_pid]
            _item: ItemsMapping = item_dict[_inst.item_id]
            _item.product_ids_set.add(_pid)
            return _item_id

        def __append_board_product__(_r, _item_id: int) -> int:
            _bid = int(_r["board_id"])
            _pid = int(_r["product_id"])
            if _pid == 0:
                return __append_board__(_r, _item_id)
            else:
                _product_inst: ProductIDMapping = product_dict.get(_pid, UNDEF_IDS)
                _board_inst: BoardIDMapping = board_dict.get(_bid, UNDEF_IDS)
                _iidx = (
                    _product_inst.item_id
                    if _product_inst != UNDEF_IDS
                    else (_board_inst.item_id if _board_inst != UNDEF_IDS else _item_id)
                )
                if _iidx == _item_id:
                    board_dict.update({_bid: BoardIDMapping(_bid, _iidx, "")})
                    product_dict.update({_pid: ProductIDMapping(_pid, _iidx, "")})
                    _inst = ItemsMapping(_iidx, "")
                    _inst.board_ids_set.add(_bid)
                    _inst.product_ids_set.add(_pid)
                    item_dict.update({_iidx: _inst})
                    _item_id += 1
                else:
                    if _product_inst == UNDEF_IDS:
                        product_dict.update(
                            {_pid: ProductIDMapping(_pid, _board_inst.item_id, "")}
                        )
                        _inst: ItemsMapping = item_dict[_board_inst.item_id]
                        _inst.product_ids_set.add(_pid)
                    else:
                        board_dict.update(
                            {_bid: BoardIDMapping(_bid, _product_inst.item_id, "")}
                        )
                        _inst: ItemsMapping = item_dict[_product_inst.item_id]
                        _inst.board_ids_set.add(_bid)
            return _item_id

        ## purchases
        # products
        for _, r in self._purchase_products_df.iterrows():
            __append_user__(r)
            item_id = __append_product__(r, item_id)
        # end : for (products)
        for _, r in self._product_scrap_df.iterrows():
            __append_user__(r)
            item_id = __append_product__(r, item_id)
        # end : for (products)

        # boards
        for _, r in self._board_owners_df.iterrows():
            __append_user__(r)
            item_id = __append_board_product__(r, item_id)
        # end : for (boards)
        for _, r in self._board_request_df.iterrows():
            __append_user__(r)
            item_id = __append_board_product__(r, item_id)
        # end : for (boards)
        for _, r in self._board_scrap_df.iterrows():
            __append_user__(r)
            item_id = __append_board_product__(r, item_id)
        # end : for (boards)

        ## likes
        for _, r in self._product_likes_df.iterrows():
            __append_user__(r)
            item_id = __append_product__(r, item_id)
        # end : for (products)
        for _, r in self._board_likes_df.iterrows():
            __append_user__(r)
            item_id = __append_board__(r, item_id)
        # end : for (boards)

        ## views
        for _, r in self._product_views_df.iterrows():
            __append_user__(r)
            item_id = __append_product__(r, item_id)
        # end : for (products)
        for _, r in self._board_views_df.iterrows():
            __append_user__(r)
            item_id = __append_board__(r, item_id)
        # end : for (boards)

        self.__dump_bins__(user_dict, "user_dict")
        self.__dump_bins__(board_dict, "board_dict")
        self.__dump_bins__(product_dict, "product_dict")
        self.__dump_bins__(item_dict, "item_dict")

        self.user_dict = user_dict
        """
        Key: user_id (int)
        Value: inst (UserEntity)
        """

        self.board_dict = board_dict
        """
        Key: board_id (int)
        Value: inst (BoardIDMapping)
        """
        self.product_dict = product_dict
        """
        Key: product_id (int)
        Value: inst (ProductIDMapping)
        """
        self.item_dict = item_dict
        """
        Key: item_id (int)
        Value: inst (ItemsMapping)
        """

    # end : private void build_item_ids()

    def __append_metadata__(self) -> None:
        """통합항목에 메타데이터를 추가하고, 항목을 여과합니다."""

        print("> append_metadata()")

        _user_dict: dict = self.__load_bins__("mapped_user_dict")
        _board_dict: dict = self.__load_bins__("mapped_board_dict")
        _product_dict: dict = self.__load_bins__("mapped_product_dict")
        _item_dict: dict = self.__load_bins__("mapped_item_dict")

        if (
            (_user_dict != None)
            and (_board_dict != None)
            and (_product_dict != None)
            and (_item_dict != None)
        ):
            self.user_dict = _user_dict
            self.item_dict = _item_dict
            self.product_dict = _product_dict
            self.board_dict = _board_dict
            return
        else:
            pass

        ## Append users interest tags
        for _, r in self._all_users_interest_tags_df.iterrows():
            user_id = int(r["user_id"])
            tag = self.__replace_tok__(r["tag"])
            tag: str = self.aliases_dict.get(tag, tag)
            if not tag in self.tags_dict:
                continue
            user: UserEntity = self.user_dict.get(user_id, UNDEF_IDS)
            if user == UNDEF_IDS:
                continue
            user.set_of_interest_tags.add(tag)
        # end : for (all_users_interest_tags)

        ## Append co-relation items contents (Board-Product)
        for _, r in self._all_board_product_df.iterrows():
            # board_id, product_id, product_name, tag_string, license_id, category_id
            board_id = int(r["board_id"])
            product_id = int(r["product_id"])
            name = str(r["product_name"]).strip()
            if name == "":
                continue
            board: BoardIDMapping = self.board_dict.get(board_id, UNDEF_IDS)
            if board != UNDEF_IDS:
                board.name = name
            product: ProductIDMapping = self.product_dict.get(product_id, UNDEF_IDS)
            if product == UNDEF_IDS:
                continue
            if product.name == "":
                product.name = name
            item: ItemsMapping = self.item_dict[product.item_id]
            for _board_id in item.board_ids_set:
                _board: BoardIDMapping = self.board_dict.get(_board_id, UNDEF_IDS)
                if _board == UNDEF_IDS:
                    continue
                if _board.name == "":
                    _board.name = name
            # end : for (mapped_board_ids)
        # end : for (forall_board-products)

        ## Append products contents
        for _, r in self._all_products_df.iterrows():
            # product_id, product_name, tag_string, license_name, license_id, category_id
            product_name = str(r["product_name"]).strip()
            if product_name == "":
                continue
            product_id = int(r["product_id"])
            if not product_id in self.product_dict:
                continue
            product: ProductIDMapping = self.product_dict[product_id]
            item: ItemsMapping = self.item_dict[product.item_id]
            if product.name == "":
                product.name = product_name
            if item.name == "":
                item.name = product_name
            for board_id in item.board_ids_set:
                board: BoardIDMapping = self.board_dict.get(board_id, UNDEF_IDS)
                if board == UNDEF_IDS:
                    continue
                if board.name == "":
                    board.name = product_name
            # end : for (board_ids)
        # end : for (all_products_info)

        ## Append boards contents
        for _, r in self._all_boards_df.iterrows():
            # board_id, user_id, title, content, tag_string, sell_request_available, created_time, modified_time, is_listed
            board_name = str(r["title"]).strip()
            if board_name == "":
                continue
            board_id = int(r["board_id"])
            board: BoardIDMapping = self.board_dict.get(board_id, UNDEF_IDS)
            if board == UNDEF_IDS:
                continue
            if board.name == "":
                board.name = board_name
            item: ItemsMapping = self.item_dict[board.item_id]
            if item.name == "":
                item.name = board_name
        # end : for (all_boards_info)

        ## Append products tags
        for _, r in self._all_product_tags_df.iterrows():
            # product_id, tag, tag_id
            product_id = int(r["product_id"])
            tag = self.__replace_tok__(r["tag"])
            tag = self.aliases_dict.get(tag, tag)
            product: ProductIDMapping = self.product_dict.get(product_id, UNDEF_IDS)
            if product == UNDEF_IDS:
                continue
            if not tag in self.tags_dict:
                continue
            product.tags.add(tag)
            item: ItemsMapping = self.item_dict[product.item_id]
            item.tags_set.add(tag)
            for board_id in item.board_ids_set:
                board: BoardIDMapping = self.board_dict.get(board_id, UNDEF_IDS)
                if board == UNDEF_IDS:
                    continue
                board.tags.add(tag)
            # end : for (board_ids)
        # end : for (all_products_tags)

        ## Append board
        for _, r in self._all_board_tags_df.iterrows():
            # board_id, tag, tag_id
            board_id = int(r["board_id"])
            board: BoardIDMapping = self.board_dict.get(board_id, UNDEF_IDS)
            if board == UNDEF_IDS:
                continue
            tag = self.__replace_tok__(r["tag"])
            tag = self.aliases_dict.get(tag, tag)
            if not tag in self.tags_dict:
                continue
            board.tags.add(tag)
            item: ItemsMapping = self.item_dict[board.item_id]
            item.tags_set.add(tag)
            for product_id in item.product_ids_set:
                product: ProductIDMapping = self.product_dict.get(product_id, UNDEF_IDS)
                if product == UNDEF_IDS:
                    continue
                product.tags.add(tag)
            # end : for (product_ids)
        # end : for (all_boards_tags)

        self.__dump_bins__(self.user_dict, "mapped_user_dict")
        self.__dump_bins__(self.board_dict, "mapped_board_dict")
        self.__dump_bins__(self.product_dict, "mapped_product_dict")
        self.__dump_bins__(self.item_dict, "mapped_item_dict")

    # end : private void append_metadata()

    def __replace_tok__(self, tok: str) -> str:
        """
        - 문자열 처리
            - 모든 공백 제거
            - 알파벳은 대문자로 치환
        """
        return tok.replace(" ", "").upper()

    # end : private str replace_tok()

    def __load_bins__(self, file_name: str) -> dict:
        """이진파일이 없으면 None을 반환합니다."""
        _file_path = f"{self._TEMP_DIR_PATH}/{file_name}.bin"
        _dict: dict = None
        if os.path.exists(_file_path):
            with open(_file_path, "rb") as fin:
                _dict: dict = pickle.load(fin)
                fin.close()
            # end : StreamWriter()
        return _dict

    # end : private dict load_bins()

    def __dump_bins__(
        self,
        obj: dict,
        file_name: str,
        target_dir_path: str = "",
    ) -> None:
        _export_dir_path = (
            self._TEMP_DIR_PATH if target_dir_path == "" else target_dir_path
        )
        _file_path = f"{_export_dir_path}/{file_name}.bin"
        with open(_file_path, "wb") as fout:
            pickle.dump(obj, fout)
            fout.close()
        # end : StreamReader()

    # end : private void dump_bins()

    def __contains_data__(self, file_name: str, sql_request: str) -> DataFrame:
        _file_path = f"{self._TEMP_DIR_PATH}/{file_name}.csv"
        df: DataFrame = None
        if os.path.exists(_file_path):
            df = pd.read_csv(_file_path)
        else:
            df = self.get_raw_data(sql_request)
            df.to_csv(_file_path)
        return df

    # end : private DataFrame contains_data()


# end : class
