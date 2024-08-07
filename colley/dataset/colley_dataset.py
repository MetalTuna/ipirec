import os

import pandas as pd

from refine import ColleyRepository, ColleyQueries
from core import BaseDataSet, DataType, TagEntity, UserEntity
from ..entity import ColleyItemEntity


class ColleyDataSet(BaseDataSet):

    def __init__(self, dataset_dir_path: str) -> None:
        super().__init__(
            dataset_dir_path=dataset_dir_path,
            data_type=DataType.E_COLLEY,
        )

    def append_interest_tags(self):
        file_path = str.format(
            "{0}/user_interest_tag_list.csv",
            self._data_root_path,
        )
        print(file_path)
        if not os.path.exists(file_path):
            ColleyRepository(
                raw_data_path=self._data_root_path,
            ).get_raw_data(
                query_str=ColleyQueries.SQL_USER_INTEREST_TAG_LIST,
            ).to_csv(path_or_buf=file_path)

        for _, r in pd.read_csv(file_path).iterrows():
            user_id = int(r["user_id"])
            tag_name = r["tag"].strip()

            if not tag_name in self.tags_dict:
                continue

            user: UserEntity = self.user_dict.get(user_id, None)
            if user == None:
                continue
            user.set_of_interest_tags.add(tag_name)
        # end : for (interest_tags_list)

    # end : public void append_interest_tags()

    def _load_metadata_(self):
        self.item_dict: dict = ColleyItemEntity.load_collection(
            dataset_dir_path=self._data_root_path
        )
        self.__read_tags__()
        # self.append_interest_tags()

    def __read_tags__(self) -> None:
        file_path = f"{self._data_root_path}/tag_list.csv"
        if not os.path.exists(file_path):
            raise FileNotFoundError()
        self.tags_dict = dict()
        """
        Key: tag_name (str)
        Value: instance (TagEntity)
        """

        for _, r in pd.read_csv(file_path).iterrows():
            tag_id = int(r["tag_id"])
            tag_name = str(r["tag"])
            inst = TagEntity(tag_id, tag_name)
            self.tags_dict.update({tag_name: inst})
        # end : for (tags_list)

        for item_id in self.item_dict.keys():
            item: ColleyItemEntity = self.item_dict[item_id]
            tags_list = list(item.tags_set)
            for tag_name in tags_list:
                # is not defined OR filtered
                if not tag_name in self.tags_dict:
                    item.tags_set.remove(tag_name)
                    continue
                tag: TagEntity = self.tags_dict[tag_name]
                tag.item_ids_set.add(item_id)
            # end : for (tags)
        # end : for (items)

    # end : private void read_tags()


# end : class
