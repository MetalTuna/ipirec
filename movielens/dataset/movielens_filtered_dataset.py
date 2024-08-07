from core import TagEntity, ItemEntity
from .movielens_dataset import MovieLensDataSet


class MovieLensFilteredDataSet(MovieLensDataSet):
    """
    - 요약:
        - 의사결정 수를 기준으로 태그들을 선별한 후, 항목들을 선별하는 모듈입니다.
    """

    def __init__(
        self,
        dataset_dir_path: str,
    ) -> None:
        super().__init__(dataset_dir_path)

    # end : init()

    def __id_index_mapping__(self) -> None:
        super().__id_index_mapping__()
        self.__tags_filtering__()

    # end : private void id_idx_mapping()

    def __tags_filtering__(self) -> None:
        """
        - 요약:
            - 각 태그에 대한 의사결정 수를 집계합니다.
            - 이 후, 태그가 없는 항목들을 제거합니다.
        """
        ## 의사결정 수를 기준으로 Top-159개의 태그를 선별(우리데이터에서의 태그수와 통일)
        sorted_tags_collection = sorted(
            self.tags_dict.values(),
            key=lambda x: x.decisions_freq_dict["total"],
            reverse=True,
        )[:159]
        tags_dict = dict()
        for inst in sorted_tags_collection:
            inst: TagEntity
            # tag_instance -> map_info_init
            inst.member_collections_init()
            tags_dict.update({inst.tag_name: inst})
        # end : for (top_n_decisioned_tags)
        self.tags_dict = tags_dict
        self.tags_count = len(sorted_tags_collection)

        item_dict = dict()
        for item_id in self.item_dict:
            item: ItemEntity = self.item_dict[item_id]
            tags_list = list(item.tags_set)
            item.tags_set.clear()
            for tag_name in tags_list:
                if not tag_name in self.tags_dict:
                    continue
                item.tags_set.add(tag_name)
                inst: TagEntity = self.tags_dict[tag_name]
                inst.item_ids_set.add(item_id)
            # end : for (T(i))

            if len(item.tags_set) == 0:
                continue
            item_dict.update({item_id: item})
        # end : for (items)
        self.item_dict = item_dict
        self.items_count = len(item_dict.keys())

        # 여과됐을 때의 관계변화를 다시 구하도록 재호출합니다.
        super().__id_index_mapping__()

    # end : private void tags_filtering()


# end : class
