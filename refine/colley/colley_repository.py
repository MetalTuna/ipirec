## Build-in LIB.
import os
import re
from abc import *

# 3rd Pty. LIB.
from tqdm import tqdm
import pickle
import pandas as pd
from pandas import DataFrame

# Custom LIB.
from core import BaseRepository, Machine
from .base_id_mapping import BaseIDMapping
from .board_id_mapping import BoardIDMapping
from .items_mapping import ItemsMapping
from .product_id_mapping import ProductIDMapping
from .colley_queries import ColleyQueries


class ColleyRepository(BaseRepository, ColleyQueries):
    """
    - 요약:
        - 상품과 게시글에 대한 통합항목을 구성합니다.
        - 상품+게시글, 상품, 게시글 순으로 항목이 구성됩니다.
    """

    def __init__(
        self,
        raw_data_path: str,
        db_src: Machine = Machine.E_MAC,
    ) -> None:
        super().__init__(raw_data_path, db_src)

        # DEF_MEM_VARs
        self.board_item_dict: dict = None
        """
        Key: board_id (int)
        Value: instance (BoardIDMapping)
        """
        self.product_item_dict: dict = None
        """
        Key: product_id (int)
        Value: instance (ProductIDMapping)
        """
        self.item_dict: dict = None
        """
        Key: mapped_item_id (int)
        Value: instance (ItemsMapping)
        """
        self.defined_tags_dict: dict = None
        """
        Key: tag_name (str)
        Value: tag_id (int)
        """

        ## DEF_CONN
        # self._conn: Connection = None
        # self._cur: DictCursor = None
        # self._access_info_dic: dict = None

        # """CONN_STR_DIC"""
        # self.__ACCESS_INFO__()

    # end : init()

    def load_data(self):
        # 원시데이터 복제
        self.__clone__()

    def dump_data(self):
        # 단, 라이센스와 카테고리 태그를 선별할 필요 없음(최종 단계에서 선별된 결과출력)
        # 의사결정 데이터 변환(고유ID -> 항목ID) => raw_dataset으로 dump,
        # 항목을 구성(상품과 게시글들을 통합하고, 항목의 태그들 추가)한 후, 저장
        self.__build_items__()
        self.__append_tags__()
        self.__dump_relations__()

    # end : public void dump_data()

    def convert_decision(self):
        """
        요약:
            diff domains items => unified domain items
            B, P and B \\cap P => I
        """

        ### [READ] 통합항목 데이타를 불러옵니다. (역 직렬화)
        # I => B, P
        file_path = f"{self._raw_data_path}/item_list.bin"
        with open(file=file_path, mode="rb") as fin:
            self.item_dict: dict = pickle.load(fin)
            fin.close()
        # end : StreamReader()

        # P => I
        file_path = f"{self._raw_data_path}/product_item.bin"
        with open(file=file_path, mode="rb") as fin:
            self.product_item_dict: dict = pickle.load(fin)
            fin.close()
        # end : StreamReader()

        # B => I
        file_path = f"{self._raw_data_path}/board_item.bin"
        with open(file=file_path, mode="rb") as fin:
            self.board_item_dict: dict = pickle.load(fin)
            fin.close()
        # end : StreamReader()

        ### [READ] 원시 의사결정 데이터를 통합항목 구성으로 사상합니다.
        self.__convert__()

    # end : convert_decision

    def __convert__(
        self,
        export_dir_path: str = "",
    ) -> None:
        """
        요약:
            의사결정 데이터의 항목번호 변환

        Args:
            export_dir_path (str, optional): 출력경로. Defaults to "".
        """
        export_dir_path = (
            self._raw_data_path if export_dir_path == "" else export_dir_path
        )
        cname_list = ["board_id", "product_id"]
        """column name list """

        # removal decisions (V, L, P)
        file_path = f"{export_dir_path}/purchase_list.csv"
        if os.path.exists(file_path):
            os.remove(file_path)
        for cname in cname_list:
            file_path = str.format(
                "{0}/{1}_list.csv",
                export_dir_path,
                cname.split("_")[0],
            )
            if os.path.exists(file_path):
                os.remove(file_path)

        files_info_list = [
            ["view_board_list.csv", 0],
            ["view_product_list.csv", 1],
            ["like_board_list.csv", 0],
            ["like_product_list.csv", 1],
            ["owned_board_product_list.csv", 1],
            ["sell_request_list.csv", 1],
            ["purchase_product_list.csv", 1],
        ]

        items_map_list = [self.board_item_dict, self.product_item_dict]
        """
        [0] board
        [1] product
        """

        export_header_str = "user_id,item_id,created_time\n"
        is_exist = False
        dtype_str = ""

        for fi_list in files_info_list:
            # [0]: file_name, [1]: toggle index of board or product
            cname: str = cname_list[fi_list[1]]
            domain_item_dict: dict = items_map_list[fi_list[1]]
            """
            Key: item_id (int)
            Value: instance (TargetIDMapping)
            """

            # [DUMP] Convert data
            dtype_str = fi_list[0].split("_")[0]
            dtype_str = (
                dtype_str if dtype_str == "view" or dtype_str == "like" else "purchase"
            )
            file_path = f"{export_dir_path}/{dtype_str}_list.csv"
            is_exist = os.path.exists(file_path)
            # 뒤에 붙이냐?
            with open(
                file=file_path,
                mode="at",
                encoding="utf-8",
            ) as fout:
                if not is_exist:
                    fout.write(export_header_str)
                # [READ] Raw data
                file_path = str.format("{0}/{1}", export_dir_path, fi_list[0])
                for _, r in pd.read_csv(file_path).iterrows():
                    user_id = int(r["user_id"])
                    domain_id = int(r[cname])
                    created_time = str(r["created_time"])
                    if not domain_id in domain_item_dict:
                        continue
                    inst: BaseIDMapping = domain_item_dict[domain_id]
                    item_id = inst.item_id
                    fout.write(f"{user_id}, {item_id},{created_time}\n")
                # end : for (target decisions)
                fout.close()
            # end : StreamWriter(AppendText)
        # end : for (DTypes)

    # end : private void convert()

    def __build_items__(self):
        """게시글과 상품이 통합된 항목을 구성합니다."""
        # board_df, board_tag_df, product_df, product_tag_df
        board_df = self.__get_df__("board_list.csv")
        product_df = self.__get_df__("product_list.csv")
        board_product_df = self.__get_df__("board_product_list.csv")

        # item_id: instance (ItemsMapping)
        item_dict = dict()
        # product_id: instance (ProductIDMapping)
        product_item_dict = dict()
        # board_id: instance (BoardIDMapping)
        board_item_dict = dict()

        ### 항목사전 만들기!
        # [PRODUCT-BOARD]
        print("ColleyRepository.build_items()\n")
        for _, r in tqdm(
            iterable=board_product_df.iterrows(),
            desc="Board-Product items",
            total=board_product_df.shape[0],
        ):
            item_id = -1
            product_id = int(r["product_id"])
            board_id = int(r["board_id"])
            product_name: str = r["product_name"]

            if product_id in product_item_dict:
                # 상품 있다
                item_id = product_item_dict[product_id].item_id

                if board_id in board_item_dict:
                    # 게시글 있다
                    inst: ItemsMapping = item_dict[item_id]
                    ## item_dict 추가
                    _item_id = board_item_dict[board_id].item_id
                    if item_id != _item_id:
                        # item_dict에 추가해주면 됨
                        inst.board_ids_set.add(board_id)
                        inst: ItemsMapping = item_dict[_item_id]
                        inst.product_ids_set.add(product_id)
                        # 병합구현 필요: 이들을 합쳐야함
                else:
                    # 게시글 없다
                    inst = BoardIDMapping(board_id, item_id, product_name)
                    board_item_dict.update({board_id: inst})
                    inst: ItemsMapping = item_dict[item_id]
                    inst.board_ids_set.add(board_id)
            else:
                # 상품 없다
                if board_id in board_item_dict:
                    # 게시글 있다
                    item_id = board_item_dict[board_id].item_id
                    inst = ProductIDMapping(product_id, item_id, product_name)
                    product_item_dict.update({product_id: inst})
                    inst: ItemsMapping = item_dict[item_id]
                    inst.product_ids_set.add(product_id)
                else:
                    # 게시글 없다
                    item_id = len(item_dict)
                    # 둘 다 추가
                    inst = BoardIDMapping(board_id, item_id, product_name)
                    board_item_dict.update({board_id: inst})
                    inst = ProductIDMapping(product_id, item_id, product_name)
                    product_item_dict.update({product_id: inst})
                    inst = ItemsMapping(item_id, product_name)
                    inst.board_ids_set.add(board_id)
                    inst.product_ids_set.add(product_id)
                    item_dict.update({item_id: inst})
        # end : for (BPs)

        # [PRODUCT]
        for _, r in tqdm(
            iterable=product_df.iterrows(),
            desc="product items",
            total=product_df.shape[0],
        ):
            product_id = int(r["product_id"])
            if product_id in product_item_dict:
                continue
            item_id = len(item_dict)
            name = r["product_name"]
            inst = ProductIDMapping(product_id, item_id, name)
            product_item_dict.update({product_id: inst})
            inst = ItemsMapping(item_id, name)
            inst.product_ids_set.add(product_id)
            item_dict.update({item_id: inst})
        # end : for (Products)

        # [BOARD]
        for _, r in tqdm(
            iterable=board_df.iterrows(),
            desc="Board items",
            total=board_df.shape[0],
        ):
            board_id = int(r["board_id"])
            if board_id in board_item_dict:
                continue
            item_id = len(item_dict)
            name = r["board_name"]
            inst = BoardIDMapping(board_id, item_id, name)
            board_item_dict.update({board_id: inst})
            inst = ItemsMapping(item_id, name)
            inst.board_ids_set.add(board_id)
            item_dict.update({item_id: inst})
        # end : for (Boards)

        # build
        self.board_item_dict = board_item_dict
        self.product_item_dict = product_item_dict
        self.item_dict = item_dict

    # end : private void build_items()

    def __append_tags__(self):
        """항목의 태그를 추가합니다."""
        print("ColleyRepository.append_tags()\n")

        self.defined_tags_dict = {
            str(r["tag"].strip()): int(r["tag_id"])
            for _, r in self.__get_df__("tag_list.csv").iterrows()
        }

        board_tag_df = self.__get_df__("board_tag_list.csv")
        product_tag_df = self.__get_df__("product_tag_list.csv")

        ### 태그를 붙이자: 상품태그를 우선으로 채우고, 상품태그가 없다면 게시글의 태그로 대체한다.
        # 상품
        for _, r in tqdm(
            iterable=product_tag_df.iterrows(),
            desc="Product tags",
            total=product_tag_df.shape[0],
        ):
            product_id = int(r["product_id"])
            if not product_id in self.product_item_dict:
                continue
            tag_name: str = r["tag"]
            product_inst: ProductIDMapping = self.product_item_dict[product_id]
            product_inst.tags.add(tag_name)
            item_inst: ItemsMapping = self.item_dict[product_inst.item_id]
            item_inst.tags_set.add(tag_name)
        # end : for (products_tags)

        # 게시글: 단, 게시글의 태그들은 상품태그에 속하는 것만 선별하는 후처리가 필요하다.
        for _, r in tqdm(
            iterable=board_tag_df.iterrows(),
            desc="Board tags",
            total=board_tag_df.shape[0],
        ):
            board_id = int(r["board_id"])
            if not board_id in self.board_item_dict:
                continue
            board_inst: BoardIDMapping = self.board_item_dict[board_id]
            # item_inst: ItemsMapping = item_dict[board_inst.item_id]
            tag_name: str = r["tag"]
            if not tag_name in self.defined_tags_dict:
                # 미등록 태그 선별
                continue
            board_inst.tags.add(tag_name)
        # end : for (boards_tags)

        for item_id in self.item_dict.keys():
            item: ItemsMapping = self.item_dict[item_id]

            # 상품의 정보로 채운다.
            for product_id in item.product_ids_set:
                inst: ProductIDMapping = self.product_item_dict[product_id]
                if item.name == "" and inst.name != "":
                    item.name = inst.name
                if len(item.tags_set) == 0 and len(inst.tags) != 0:
                    item.tags_set = inst.tags
                if item.name != "" and len(item.tags_set) != 0:
                    break
            # end : for (P(i))
            if item.name != "" and len(item.tags_set) != 0:
                continue

            # 상품의 정보가 없다면, 게시글의 정보로 채운다.
            for board_id in item.board_ids_set:
                inst: BoardIDMapping = self.board_item_dict[board_id]
                if item.name == "" and inst.name != "":
                    item.name = inst.name
                if len(item.tags_set) == 0 and len(inst.tags) != 0:
                    item.tags_set = inst.tags
                if item.name != "" and len(item.tags_set) != 0:
                    break
            # end : for (B(i))
        # end : for (items)

    # end: private void append_tags()

    def __dump_relations__(self):
        """항목관련 변수들을 저장합니다."""
        file_name_list = ["board_item", "product_item", "item_list"]
        obj_list = [self.board_item_dict, self.product_item_dict, self.item_dict]

        for idx in range(len(file_name_list)):
            obj = obj_list[idx]
            # [BINARY]
            file_path = str.format(
                "{0}/{1}.bin", self._raw_data_path, file_name_list[idx]
            )
            with open(file=file_path, mode="wb") as fout:
                pickle.dump(obj=obj, file=fout)
                fout.close()
            # end : StreamWriter()

        # end : for (files)

    # end : private void dump_relations()

    def __get_df__(
        self,
        file_name: str,
    ) -> DataFrame:
        """csv -> df"""
        return pd.read_csv(f"{self._raw_data_path}/{file_name}")

    # end : get_df()

    def __tags_list_redefine__(
        self,
        df: DataFrame,
        dest_file_path: str,
    ) -> None:
        """
        요약:
            태그목록 재처리를 위한 함수입니다.

        Args:
            df (DataFrame): tag_list_df
            dest_file_path (str): dump path

        Raises:
            NotImplementedError: _description_
        """
        dict_of_tags_info = dict()
        """
        Key: tag_name
        Value: "tag_id", "tag_idx": {
            value (int)
        }
        """
        tidx = 0
        contains_rm_tok = False
        set_of_removal_tokens = {
            "챌린지",
        }

        for _, r in df.iterrows():
            # DB에 정의된 ID -> tag_id
            tag_id = int(r["tag_id"])
            tag_name = r["tag"]
            contains_rm_tok = False
            for rm_tok in set_of_removal_tokens:
                contains_rm_tok = (
                    re.match(
                        f".*{rm_tok}.*",
                        tag_name,
                    )
                    != None
                )
                if contains_rm_tok:
                    break
            # end : for (removal_tokens)
            if contains_rm_tok:
                continue
            dict_of_tags_info.update(
                {
                    tag_name: {
                        "tag_id": tag_id,
                        "tag_idx": tidx,
                    }
                }
            )
            tidx += 1
        # end : for (tags_list)

        # [DUMP] tags_list
        with open(
            file=dest_file_path,
            mode="wt",
            encoding="utf-8",
        ) as fout:
            # FILE_HEADER: Column families...
            line = ",tag_id,tag\n"
            fout.write(line)
            for tag_name in dict_of_tags_info.keys():
                inst: dict = dict_of_tags_info[tag_name]
                # [FORMAT] tag_idx (int), tag_id (int), tag (str)
                line = str.format(
                    "{0},{1},{2}\n",
                    inst["tag_idx"],
                    inst["tag_id"],
                    tag_name,
                )
                fout.write(line)
            # end : for (tags_list)
            fout.close()
        # end : StreamWriter()

    # end : private void tags_list_redefine()

    def __clone__(self) -> None:
        """DB -> local csv"""
        if not os.path.exists(self._raw_data_path):
            os.makedirs(self._raw_data_path)

        print("DB to CSV.")

        # 사용자
        self.get_raw_data(self.SQL_USER_LIST).to_csv(
            f"{self._raw_data_path}/user_list.csv"
        )
        # 게시글
        df = self.get_raw_data(self.SQL_BOARD_LIST)
        df.to_csv(f"{self._raw_data_path}/board_list_raw.csv")
        df.rename(columns={"title": "board_name"}).to_csv(
            f"{self._raw_data_path}/board_list.csv"
        )
        # self.__repo_df__(self.SQL_BOARD_LIST).to_csv( f"{self._raw_data_path}/board_list.csv" )
        # 상품
        self.get_raw_data(self.SQL_PRODUCT_LIST).to_csv(
            f"{self._raw_data_path}/product_list.csv"
        )
        # 게시글-상품 관계
        self.get_raw_data(self.SQL_BOARD_PRODUCT_LIST).to_csv(
            f"{self._raw_data_path}/board_product_list.csv"
        )
        # self.__repo_df__(self.SQL_DEFINED_TAG_LIST).to_csv(f"{self._raw_data_path}/tag_list.csv")
        # 태그
        tag_list_df = self.get_raw_data(
            self.SQL_DEFINED_TAG_LIST,
        )
        self.__tags_list_redefine__(
            df=tag_list_df,
            dest_file_path=f"{self._raw_data_path}/tag_list.csv",
        )
        self.get_raw_data(self.SQL_USER_INTEREST_TAG_LIST).to_csv(
            f"{self._raw_data_path}/user_interest_tag_list.csv"
        )
        self.get_raw_data(self.SQL_BOARD_TAG_LIST).to_csv(
            f"{self._raw_data_path}/board_tag_list.csv"
        )
        self.get_raw_data(self.SQL_PRODUCT_TAG_LIST).to_csv(
            f"{self._raw_data_path}/product_tag_list.csv"
        )
        self.get_raw_data(self.SQL_LICENSE_TAG_LIST).to_csv(
            f"{self._raw_data_path}/license_tag_list.csv"
        )
        self.get_raw_data(self.SQL_CATEGORY_TAG_LIST).to_csv(
            f"{self._raw_data_path}/category_tag_list.csv"
        )
        ## 의사결정
        # 봤다
        self.get_raw_data(self.SQL_OPEN_BOARD_LIST).to_csv(
            f"{self._raw_data_path}/view_board_list.csv"
        )
        self.get_raw_data(self.SQL_OPEN_PRODUCT_LIST).to_csv(
            f"{self._raw_data_path}/view_product_list.csv"
        )
        # 좋다
        self.get_raw_data(self.SQL_LIKE_BOARD_LIST).to_csv(
            f"{self._raw_data_path}/like_board_list.csv"
        )
        self.get_raw_data(self.SQL_LIKE_PRODUCT_LIST).to_csv(
            f"{self._raw_data_path}/like_product_list.csv"
        )
        # 판매요청
        self.get_raw_data(self.SQL_REQUEST_LIST).to_csv(
            f"{self._raw_data_path}/sell_request_list.csv"
        )
        # 샀다
        self.get_raw_data(self.SQL_OWNED_USER_LIST).to_csv(
            f"{self._raw_data_path}/owned_board_product_list.csv"
        )
        self.get_raw_data(self.SQL_PURCHASE_LIST).to_csv(
            f"{self._raw_data_path}/purchase_product_list.csv"
        )

    # end : private void clone()


# end : class
