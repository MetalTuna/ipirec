import os
import pickle

from core import ItemEntity
from refine import ItemsMapping


class ColleyItemEntity(ItemEntity):

    def __init__(
        self,
        item_id: int = -1,
        tags_set: set = set(),
        name: str = "",
        board_ids: set = set(),
        product_ids: set = set(),
    ):
        super().__init__(item_id, tags_set)
        self.item_name = name
        self.board_ids = board_ids
        self.product_ids = product_ids

    @staticmethod
    def load_collection(dataset_dir_path: str) -> dict:
        """_summary_

        Args:
            dataset_dir_path (str): ${DATA_HOME}

        Returns:
            dict:
            Key: item_id (int)
            Value: instance (ItemsMapping)
        """

        # dir_path /Users/taegyu.hwang/Documents/tghwang_git_repo/ipitems_analysis/data/colley
        # files = [view, like, purchase]_list.csv, item_list.bin, board_item.bin, product_item.bin

        ### raw_instances => colley_item_instance
        # 원시데이터 항목을 콜리 항목으로 변환
        file_path = f"{dataset_dir_path}/item_list.bin"
        _item_dict: dict = None
        if not os.path.exists(file_path):
            raise FileNotFoundError()
        with open(file=file_path, mode="rb") as fin:
            _item_dict: dict = pickle.load(fin)
            fin.close()
        # end : StreamReader()
        if _item_dict == None:
            raise NotImplementedError()

        item_dict = dict()
        for item_id in _item_dict.keys():
            _inst: ItemsMapping = _item_dict[item_id]
            inst = ColleyItemEntity(
                item_id=_inst.item_id,
                tags_set=_inst.tags_set,
                name=_inst.name,
                board_ids=_inst.board_ids_set,
                product_ids=_inst.product_ids_set,
            )
            item_dict.update({inst.item_id: inst})
        return item_dict

    @staticmethod
    def parse_entity(iter):
        """
        요약:
            사용하지 않는 함수

        Raises:
            ArithmeticError: _description_
        """
        print("사용하지 않음")
        raise ArithmeticError()


# end : class
