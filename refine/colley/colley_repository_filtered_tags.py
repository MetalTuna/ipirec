"""
- 요약:
    - 실험용 데이터 셋 구성과정에서, 태그들을 선별하기 위한 모듈입니다.
    - 사용하지 않습니다. (24.04.14)
"""

import pickle
import pandas as pd
from tqdm import tqdm

from core import BaseAction
from .board_id_mapping import BoardIDMapping
from .colley_repository import ColleyRepository
from .items_mapping import ItemsMapping
from .product_id_mapping import ProductIDMapping


class ColleyRepositoryFilteredTags(ColleyRepository):
    """
    - 요약:
        - 태그의 규모축소를 통한 데이터 셋을 생성하는 클래스입니다.
    """

    def __init__(
        self,
        raw_data_path: str,
    ) -> None:
        super().__init__(raw_data_path)
        self._EXPORT_DIR_PATH = self._raw_data_path
        self.__LEMMA_DICT_DIR_PATH = "/Users/taegyu.hwang/Documents/tghwang_git_repo/ipitems_analysis/resources/tags_dictionary"
        self.lemma_tags_dict: dict = None
        """
        - 요약:
            - 태그들의 통합사전입니다. 
            
        >>> `alias.csv`와 `synonym.csv`로 생성됩니다. 
        
        - 구성:
            - Key: to (str)
            - Value: from (str) - x,y,z, ...
        """
        self.lemma_tags_set: set = None
        """
        - 요약:
            - 변환된 태그들의 집합입니다.
            
        - 용도:
            - 이 변수를 참조해서 원시 태그들을 선별합니다.
                - 있으면 사용, 없으면 제거
        """

    # end : init()

    def dump_data(self):
        self.__build_items__()
        self.__append_tags__()
        self.__tags_filtering__()
        self.__dump_relations__()

    # end : public override void dump_data()

    def __tags_filtering__(self):
        """태그 선별을 위해 추가된 함수입니다."""
        # 사전을 불러온다.
        self.__load_lemma_dictionary__()
        ### 동의어와 통합사전을 사용해서 태그들을 통합한다.
        self.__tags_lemmatization__()

    # end : private void tags_filtering()

    def __convert__(self, export_dir_path: str = "") -> None:
        # 태그에 속하는 사용자, 항목의 빈도 수를 구하고, 빈도수를 기준으로 태그들을 선별한다.
        # 이 후, 태그가 없는 항목들을 제거한다.
        self.__tags_frequencies__()
        super().__convert__(export_dir_path)
        # 태그가 없는 항목들을 제거해서 의사결정 데이터를 재처리한 후, 사용자들을 다시 선별한다.
        self.__refine_decisions__()

    # end : private override void convert()

    def __load_lemma_dictionary__(self):
        """사전을 불러온다."""
        # 통합 사전 (alias.csv)
        self.lemma_tags_set = set()
        self.lemma_tags_dict = dict()

        file_path = f"{self.__LEMMA_DICT_DIR_PATH}/alias.csv"
        for _, r in pd.read_csv(file_path).iterrows():
            lemma = r["to"].strip()
            self.lemma_tags_set.add(lemma)
            for tok in r["from"].split(","):
                tok = tok.strip()
                if not tok in self.lemma_tags_dict:
                    self.lemma_tags_dict.update({tok: lemma})
            # end : for (aliases)
        # end : for (alises)

        # 동의어 사전 (alias.csv)
        file_path = f"{self.__LEMMA_DICT_DIR_PATH}/synonym.csv"
        for _, r in pd.read_csv(file_path).iterrows():
            lemma = r["to"].strip()
            self.lemma_tags_set.add(lemma)
            for tok in r["from"].split(","):
                tok = tok.strip()
                if not tok in self.lemma_tags_dict:
                    self.lemma_tags_dict.update({tok: lemma})
            # end : for (aliases)
        # end : for (alises)

        file_path = f"{self._EXPORT_DIR_PATH}/lemma_tags_list.csv"
        with open(
            file=file_path,
            mode="wt",
            encoding="utf-8",
        ) as fout:
            fout.write("tag_id.,tag\n")
            line_no = 1
            for tag_name in self.lemma_tags_set:
                fout.write(f"{line_no},{tag_name}\n")
                line_no += 1
            fout.close()
        # end : StreamWriter()

    # end : private void load_lemma_dictionary()

    def __tags_lemmatization__(self):
        """통합사전으로 항목의 태그들을 선별한다."""
        for item_id in self.item_dict.keys():
            item: ItemsMapping = self.item_dict[item_id]
            tags_list = list(item.tags_set)
            item.tags_set.clear()

            for tag_name in tags_list:
                tag_name = self.lemma_tags_dict.get(tag_name, tag_name)
                if not tag_name in self.lemma_tags_set:
                    continue
                item.tags_set.add(tag_name)
            # end : for (tags)
        # end : for (items)

    # end : private void tags_lemmatization()

    def __tags_frequencies__(self):
        """태그에 참여한 사용자와 항목의 빈도수를 구한 후, 태그들을 선별한다."""
        item_ids = list(self.item_dict.keys())
        for item_id in item_ids:
            item: ItemsMapping = self.item_dict[item_id]
            # 태그가 없는 항목 제거
            if len(item.tags_set) == 0:
                for board_id in item.board_ids_set:
                    if board_id in self.board_item_dict:
                        self.board_item_dict.pop(board_id)
                # end : for (boards)
                for product_id in item.product_ids_set:
                    if product_id in self.product_item_dict:
                        self.product_item_dict.pop(product_id)
                # end : for (products)
                self.item_dict.pop(item_id)
            # end : if (|T(i)| == 0)
        # end : for (items)

        ### [EXPORT] filtered_items
        # [BIN] board_item
        file_path = f"{self._EXPORT_DIR_PATH}/board_item.bin"
        with open(file=file_path, mode="wb") as fout:
            pickle.dump(self.board_item_dict, fout)
            fout.close()
        # end : StreamWriter()

        # [BIN] item_list
        file_path = f"{self._EXPORT_DIR_PATH}/item_list.bin"
        with open(file=file_path, mode="wb") as fout:
            pickle.dump(self.item_dict, fout)
            fout.close()
        # end : StreamWriter()

        # [BIN] product_item
        file_path = f"{self._EXPORT_DIR_PATH}/product_item.bin"
        with open(file=file_path, mode="wb") as fout:
            pickle.dump(self.product_item_dict, fout)
            fout.close()
        # end : StreamWriter()

    # end : private void tags_frequencies()

    def __refine_decisions__(self):
        """태그가 없는 항목들을 제거하고, 의사결정 데이터를 재 처리한 후, 여기에 속한 사용자들을 재선별한다."""
        export_header_str = "user_id,item_id,created_time\n"
        decision_type_list = ["view", "like", "purchase"]

        for d_type in decision_type_list:
            file_path = f"{self._EXPORT_DIR_PATH}/{d_type}_list.csv"
            raw_insts = BaseAction.load_collection(file_path)
            with open(
                file=file_path,
                mode="wt",
                encoding="utf-8",
            ) as fout:
                fout.write(export_header_str)
                for inst in raw_insts:
                    inst: BaseAction
                    if not inst.item_id in self.item_dict:
                        continue
                    # end : if (|T(i)| == 0)
                    fout.write(f"{inst.user_id},{inst.item_id},{inst.created_time}\n")
                # end : for (decisions)
                fout.close()
            # end : StreamWriter()

        # end : for (decision_types)

    # end : private void refine_decisions()

    def __build_items__(self):
        """게시글과 상품이 통합된 항목을 구성합니다."""
        # board_df, board_tag_df, product_df, product_tag_df
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

        # build
        self.board_item_dict = board_item_dict
        self.product_item_dict = product_item_dict
        self.item_dict = item_dict

    # end : private override void build_items()


# end : class
