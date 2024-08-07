from tqdm import tqdm

from core import Machine
from .board_id_mapping import BoardIDMapping
from .product_id_mapping import ProductIDMapping
from .items_mapping import ItemsMapping
from .colley_repository import ColleyRepository


class ColleyRepositoryProductItems(ColleyRepository):
    """상품을 중심으로 통합된 항목을 구성합니다. (상품+게시글, 상품)"""

    def __init__(
        self,
        raw_data_path: str,
        db_src: Machine = Machine.E_MAC,
    ) -> None:
        super().__init__(raw_data_path, db_src)

    # end : init()

    def __build_items__(self):
        """상품을 중심으로 통합된 항목을 구성합니다. (상품+게시글, 상품)"""
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

        # build items
        self.board_item_dict = board_item_dict
        self.product_item_dict = product_item_dict
        self.item_dict = item_dict

    # end : private void build_items()


# end : class
