from core import BaseDataSet, DataType, TagEntity
from ..entity import MovieLensItemEntity


class MovieLensDataSet(BaseDataSet):
    def __init__(self, dataset_dir_path: str) -> None:
        super().__init__(
            dataset_dir_path=dataset_dir_path,
            data_type=DataType.E_MOVIELENS,
        )

    def _load_metadata_(self) -> None:
        # movies.csv + item_list.csv => item_dict
        self.item_dict = MovieLensItemEntity.load_collection(
            dataset_dir_path=self._data_root_path
        )
        self.__read_tags__()

    def __read_tags__(self) -> None:
        self.tags_dict = dict()
        tidx = 0
        for item_id in self.item_dict.keys():
            inst: MovieLensItemEntity = self.item_dict[item_id]
            for tag_name in inst.tags_set:
                if not tag_name in self.tags_dict:
                    tag = TagEntity(tidx, tag_name)
                    self.tags_dict.update({tag_name: tag})
                    tidx += 1
                tag: TagEntity = self.tags_dict[tag_name]
                tag.item_ids_set.add(item_id)
                # self.tags_dict.update({tag_name: tag})
            # end : for (tags)
        # end : for (items)

    # end : private void read_tags()


# end : class
