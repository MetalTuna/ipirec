import os

import pandas as pd


from core import ItemEntity, TagEntity
from .colley_dataset import ColleyDataSet


class ColleyFilteredDataSet(ColleyDataSet):
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

    def _load_metadata_(self) -> None:
        super()._load_metadata_()
        self.__load_lemma_dictionary__()
        self.__tags_filtering__()
        # self.append_interest_tags()

    # end : protected void load_metadata()

    def __tags_filtering__(self) -> None:
        """
        - 요약:
            - 태그들의 표제어 처리관련 기능입니다.
        - 작업:
            - 태그사전에 대한 표제어 처리를 진행합니다.
            - 항목이 속한 태그들의 표제어 처리를 진행합니다.
            - 태그가 없는 항목들을 제거합니다. (여기 미구현)
        """

        # 태그사전을 표제어 처리
        filtered_tags_dict = dict()
        self.tags_count = 0
        for tag_name in self.tags_dict.keys():
            inst: TagEntity = self.tags_dict[tag_name]
            if tag_name in self.__lemma_tags_set:
                filtered_tags_dict.update({tag_name: inst})
                self.tags_count += 1
            # end : if (is_lemma_tok?)
        # end : for (tags)
        self.tags_dict = filtered_tags_dict

        # 항목이 속한 태그에 대한 표제어 처리
        filtered_items_dict = dict()
        self.items_count = 0
        for item_id in self.item_dict.keys():
            item: ItemEntity = self.item_dict[item_id]
            tags_list = list(item.tags_set)
            item.tags_set.clear()
            for tag_name in tags_list:
                tok = self.__aliase_dict.get(tag_name, tag_name)
                if tok in self.__lemma_tags_set:
                    item.tags_set.add(tok)
                # end : if (is_lemma_tok?)
            # end : for (T(i))

            if len(item.tags_set) == 0:
                continue
            filtered_items_dict.update({item_id: item})
            self.items_count += 1
        # end : for (items)
        self.item_dict = filtered_items_dict

    # end : private void tags_filtering()

    def __load_lemma_dictionary__(self) -> None:
        """
        - 요약:
            - 표제어 사전을 불러옵니다.
            - alias.csv, synonym.csv
        """
        file_dir_path = self._data_root_path.replace(
            "data/colley", "resources/tags_dictionary"
        )
        if not os.path.exists(file_dir_path):
            raise NotADirectoryError()

        ## alias, synonym
        self.__aliase_dict = dict()
        self.__lemma_tags_set = set()

        file_path = f"{file_dir_path}/alias.csv"
        for _, r in pd.read_csv(file_path).iterrows():
            lemma_tok = r["to"].strip()
            for tok in r["from"].split(","):
                tok = tok.strip()
                if not tok in self.__aliase_dict:
                    self.__aliase_dict.update({tok: lemma_tok})
                    self.__lemma_tags_set.add(lemma_tok)
                # end : if (contains_tok?)
            # end : for (aliases)
        # end : for (alias)

        file_path = f"{file_dir_path}/synonym.csv"
        for _, r in pd.read_csv(file_path).iterrows():
            lemma_tok = r["to"].strip()
            for tok in r["from"].split(","):
                tok = tok.strip()
                if not tok in self.__aliase_dict:
                    self.__aliase_dict.update({tok: lemma_tok})
                    self.__lemma_tags_set.add(lemma_tok)
                # end : if (contains_tok?)
            # end : for (synonym_toks)
        # end : for (synonym)

    # end : private void load_lemma_dictionary()


# end : class
