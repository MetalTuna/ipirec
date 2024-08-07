"""
[작성일] 24.07.10 14:45. 사용자와 항목이 속한 태그들의 색인을 전담하도록 구조를 변경함
"""

import os

import pandas as pd

from core import UserEntity, ItemEntity, TagEntity, BaseAction
from ..entity import ColleyItemEntity
from .colley_dataset import ColleyDataSet


class ColleyDataSetRev(ColleyDataSet):

    def __init__(self, dataset_dir_path: str) -> None:
        self.user_id_to_tag_idx_dict: dict = None
        """
        - 요약:
            - 사용자가 속하는 태그들의 색인사전 (모든 의사결정된 항목들의 태그들, 관심태그들)
        - 구조:
            - Key: user_id (int)
            - Value: tags_idxs (set)
        """
        self.item_id_to_tag_idx_dict: dict = None
        """
        - 요약:
            - 항목이 속하는 태그들의 색인사전
        - 구조:
            - Key: user_id (int)
            - Value: tags_idxs (set)
        """
        # ColleyDataSet.__init__(self, dataset_dir_path)
        super().__init__(dataset_dir_path)

    # end : init()

    def __id_index_mapping__(self) -> None:
        """
        - 요약:
            - 사용자, 항목, 태그들을 색인합니다. (indexing)
            - 사용자, 항목, 태그의 메타데이터에 따른 관계를 표현합니다. (features relations mapping)
            - 의사결정 내역에 대한 관계들을 표현합니다. (decisions relations mapping)
        """
        self.user_id_to_idx = dict()
        self.user_idx_to_id = dict()
        self.item_id_to_idx = dict()
        self.item_idx_to_id = dict()
        self.tag_name_to_idx = dict()
        self.tag_idx_to_name = dict()

        iidx = 0
        uidx = len(self.user_dict) if self.__read_users_list__() else 0

        for tag_name in self.tags_dict.keys():
            tag: TagEntity = self.tags_dict.get(tag_name)
            if tag_name in self.tag_name_to_idx:
                continue
            self.tag_name_to_idx.update({tag_name: tag._idx})
            self.tag_idx_to_name.update({tag._idx: tag_name})
        # end : for (tags)

        # item tags reallocation
        for item_id in self.item_dict.keys():
            item: ItemEntity = self.item_dict.get(item_id)
            tags_list = list(item.tags_set)
            item.tags_set.clear()
            for tag_name in tags_list:
                if not tag_name in self.tags_dict:
                    continue
                # end : if (is_filtered_tag?)
                item.tags_set.add(tag_name)
            # end : for (T(i))
        # end : for (items)

        # 의사결정 내역을 추가합니다.
        decision_dict = dict()
        for kwd in self._decision_dict.keys():
            decisions_list = list()
            for inst in self._decision_dict[kwd]:
                inst: BaseAction
                user_id = inst.user_id
                item_id = inst.item_id

                # is_filtered_items
                if not item_id in self.item_dict:
                    continue
                decisions_list.append(inst)
                if not user_id in self.user_id_to_idx:
                    self.user_id_to_idx.update({user_id: uidx})
                    self.user_idx_to_id.update({uidx: user_id})
                    self.user_dict.update(
                        {
                            inst.user_id: UserEntity(
                                user_idx=uidx,
                                user_id=user_id,
                            )
                        }
                    )
                    uidx += 1
                if not item_id in self.item_id_to_idx:
                    self.item_id_to_idx.update({item_id: iidx})
                    self.item_idx_to_id.update({iidx: item_id})
                    iidx += 1
                user: UserEntity = self.user_dict.get(user_id)
                item: ItemEntity = self.item_dict.get(item_id)
                user.dict_of_decision_item_ids[kwd].add(item_id)
                item.dict_of_users_decision[kwd].add(user_id)

                for tag_name in item.tags_set:
                    # if not tag_name in self.tags_dict:
                    #    continue
                    tag: TagEntity = self.tags_dict.get(tag_name)
                    tag.user_ids_set.add(user_id)
                    user.dict_of_interaction_tags[kwd].add(tag_name)
                    tag.decisions_freq_dict[kwd] += 1
                    tag.decisions_freq_dict["total"] += 1
                # end : for (T(i))
                # self._decison_dict[kwd] = decisions_list
                decision_dict.update({kwd: decisions_list})
            # end : for (decisions_list)
        # end : for (decision_types)
        self._decision_dict.clear()
        self._decision_dict = decision_dict
        self.users_count = uidx
        self.items_count = iidx
        self.tags_count = len(self.tags_dict)

        print(
            str.format(
                "[INFO]\n- {0:9s}: {1:9d}\n- {2:9s}: {3:9d}\n- {4:9s}: {5:9d}",
                "Users",
                self.users_count,
                "Items",
                self.items_count,
                "Tags",
                self.tags_count,
            )
        )
        for kwd in decision_dict.keys():
            print(str.format("- {0:9s}: {1:9d}", kwd, len(decision_dict[kwd])))
        # end : for (decision_types)
        self.__tags_indexing__()

    # end : private override void id_index_mapping()

    def __tags_indexing__(self) -> None:
        user_tags_idx_dict = dict()
        for user_id in self.user_dict.keys():
            user: UserEntity = self.user_dict.get(user_id)
            _tags_set = {
                _tag
                for _, _tags in user.dict_of_interaction_tags.items()
                for _tag in _tags
            }.union({_tag for _tag in user.set_of_interest_tags})
            user.dict_of_interaction_tags.update({"all": _tags_set})
            user_tags_idx_dict.update(
                {
                    user_id: {
                        self.tag_name_to_idx[_tag]
                        for _tag in _tags_set
                        if _tag in self.tag_name_to_idx
                    }
                }
            )
        # end : for (users)

        self.item_id_to_tag_idx_dict = {
            item_id: {
                self.tag_name_to_idx[_tag]
                for _tag in item.tags_set
                if _tag in self.tag_name_to_idx
            }
            for item_id, item in self.item_dict.items()
        }
        self.user_id_to_tag_idx_dict = user_tags_idx_dict

    # end : private void tags_indexing()

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
            inst = TagEntity(tag_id, tag_name, int(_))
            self.tags_dict.update({tag_name: inst})
        # end : for (tags_list)

        for item_id in self.item_dict.keys():
            item: ColleyItemEntity = self.item_dict.get(item_id)
            tags_list = list(item.tags_set)
            for tag_name in tags_list:
                # is not defined OR filtered
                if not tag_name in self.tags_dict:
                    item.tags_set.remove(tag_name)
                    continue
                tag: TagEntity = self.tags_dict.get(tag_name)
                tag.item_ids_set.add(item_id)
            # end : for (tags)

    # end : private override void read_tags()


# end : class
