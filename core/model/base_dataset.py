import os
from abc import *
from configparser import ConfigParser
import pickle

from ..defines import DataType, DecisionType
from ..entity import BaseAction, UserEntity, ItemEntity, TagEntity
from ..io import InstanceIO


# class BaseDataSet(metaclass=ABCMeta):
class BaseDataSet(InstanceIO):
    """
    - 요약:
        - 데이터 분석을 위해, 데이터 셋을 적재와 접근을 관리하는 추상 클래스입니다.
        - 이 추상 클래스를 상속해서 분석에 사용할 데이터 셋을 정의하세요.
        - 이 클래스를 바탕으로 아래의 두 데이터 셋이 정의됐습니다.
            - MovieLensDataSet: `lc_corr.dataset.movielens_dataset.py`
            - ColleyDataSet: `lc_corr.dataset.colley_dataset.py`
    ----
    - 구성:
        - 멤버변수:
            - decisions_list: [view, like, purchase]_list
            - vector_id_to_idx_map: [user, item, tag]_[id, idx]_to_[idx, id] (dict)
            - [user, item, tag]s_count (int)
        - 멤버함수
            - public void load_dataset(): 데이터를 불러옵니다. (메타데이터 -> 의사결정 -> 매핑)
            - public void append_decisions(): 임의로 의사결정 내역을 등록하거나 기존 내역에 추가합니다.
                - 이 함수를 호출한다면, 의사결정 관계를 다시 계산하세요.
                ====
                    - relations_mapping_init(): 관계표현 초기화
                    - id_index_mapping(): 벡터 색인, 벡터 매핑, 의사결정 매핑
            - public void load_decisions(): 의사결정 데이터를 불러옵니다.
            - public void relations_mapping_init(): 의사결정에 따른 사용자, 항목, 태그들의 관계표현을 초기화합니다.
            - private void id_index_mapping(): 벡터(사용자, 항목, 태그)를 색인하고(indexing), 특징 간의 관계와 의사결정에 대한 관계를 사상합니다(mapping).
            - protected void dump_dir_path_conf()
        - 추상함수
            - protected abstract void load_metadata()
    ----
    >>>  dataset = ColleyDataSet(dataset_dir_path, DataType.`E_COLLEY`)
    """

    def __init__(
        self,
        dataset_dir_path: str,
        data_type: DataType,
        is_debug_mode: bool = True,
    ) -> None:
        self._config_info: ConfigParser = None
        """
        - 요약:
            - 모델변수 파악을 위한 변수입니다.
            - 이 클래스에서는 데이터 셋 종류만 담습니다. (MovieLens OR Colley)
            - 이 클래스 인스턴스에 모델, 추정기의 변수들이 덧붙여집니다.
        """
        self._data_type = data_type
        self.IS_DEBUG_MODE = is_debug_mode
        self._data_root_path = dataset_dir_path
        self._dump_dir_path = self._dump_dir_path_conf_()
        if (not os.path.exists(self._data_root_path)) or (
            not os.path.isdir(self._data_root_path)
        ):
            raise NotADirectoryError()
        self._decision_dict = {
            "view": list(),
            "like": list(),
            "purchase": list(),
        }
        self.user_id_to_idx: dict = None
        self.user_idx_to_id: dict = None
        self.item_id_to_idx: dict = None
        self.item_idx_to_id: dict = None
        self.tag_name_to_idx: dict = None
        """
        Key: tag_name (str)
        Value: tag_idx (int)
        """
        self.tag_idx_to_name: dict = None
        """
        Key: tag_idx (int)
        Value: tag_name (str)
        """
        self.user_dict = dict()
        """
        Key: user_id (int)
        Value: instance (UserEntity)
        """
        self.item_dict: dict = None
        """
        Key: item_id (int)
        Value: instance (ItemEntity)
        """
        self.tags_dict: dict = None
        """
        Key: tag_name (str)
        Value: instance (TagEntity)
        """
        self.users_count: int = -1
        self.items_count: int = -1
        self.tags_count: int = -1

        ## Config..
        self._config_info = ConfigParser()
        __sec = "DataSet"
        __opt = "name"
        self._config_info.add_section(section=__sec)
        self._config_info.set(
            section=__sec,
            option=__opt,
            value=DataType.to_str(data_type),
        )

    # end : init()

    def kfold_file_path(
        self,
        kfold_set_no: int,
        decision_type: DecisionType,
        is_train_set: bool = True,
    ) -> str:
        target_kwd_str = "train" if is_train_set else "test"
        return str.format(
            "{0}/{1}_{2}_{3}_list.csv",
            self._data_root_path,
            target_kwd_str,
            kfold_set_no,
            DecisionType.to_str(decision_type),
        )

    def load_kfold_train_set(
        self,
        kfold_set_no: int,
    ) -> None:
        """
        - 요약:
            - 정해진 교차검증 집합의 훈련데이터를 불러옵니다.

        - 매개변수:
            - kfold_set_no (int): 교차검증 집합 번호입니다.
        """
        self._config_info.set(
            section="DataSet",
            option="kfold_set_no",
            value=str(kfold_set_no),
        )
        self._load_metadata_()
        for decision_type in DecisionType:
            file_path = self.kfold_file_path(
                kfold_set_no=kfold_set_no,
                decision_type=decision_type,
            )
            """
            file_path = str.format(
                "{0}/train_{1}_{2}_list.csv",
                self._data_root_path,
                kfold_set_no,
                DecisionType.to_str(decision_type),
            )
            """
            self.append_decisions(
                file_path=file_path,
                decision_type=decision_type,
            )
        # end : for (decision_types)
        self.__id_index_mapping__()
        print(
            str.format(
                "[INFO]\n- {0:5s}: {1:9d}\n- {2:5s}: {3:9d}\n- {4:5s}: {5:9d}\n",
                "Users",
                self.users_count,
                "Items",
                self.items_count,
                "Tags",
                self.tags_count,
            )
        )

    # end : public void load_kfold_train_set()

    def load_dataset(
        self,
    ):
        """
        - 요약:
            - 데이터 셋을 불러옵니다.
        - 호출 순서
            - load_metadata()
            - load_decisions()
            - id_index_mapping()
        """
        print("[READ] Dataset.load_dataset()")
        self._load_metadata_()
        self._load_decisions_()
        # self.__items_recallocation__()
        self.__id_index_mapping__()
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

    # end : public void load_dataset()

    def append_decisions(
        self,
        file_path: str,
        decision_type: DecisionType = DecisionType.E_VIEW,
    ) -> None:
        """
        - 요약:
            - 임의로 의사결정 내역을 등록하거나 기존 내역에 추가합니다.

        - 매개변수:
            - file_path (str): 추가할 의사결정 파일의 경로
            - decision_type (DecisionType, optional): 여기에서 설정된 의사결정 타입에 추가합니다.
                - ex. DecisionType.E_VIEW이면, 현재 의사결정 구성의 열람내역에 추가됩니다.

        - 예외:
            - ValueError: 미정의된 의사결정 타입을 사용하면 예외발생
            - FileNotFoundError: 의사결정 파일 없으면 예외발생
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError()
        kwd = ""
        match (decision_type):
            case DecisionType.E_VIEW:
                kwd = "view"
            case DecisionType.E_LIKE:
                kwd = "like"
            case DecisionType.E_PURCHASE:
                kwd = "purchase"
            case _:
                raise ValueError()
        decision_list: list = BaseAction.load_collection(file_path)
        for inst in decision_list:
            self._decision_dict[kwd].append(inst)
        # end : for (decisions)

    # end : public void append_decisions()

    # '''
    def _dump_dir_path_conf_(
        self,
        target_path: str = "",
        path_sep: str = "/",
    ) -> str:
        """
        요약:
            멤버변수가 저장될 폴더경로를 처리합니다.

        Args:
            target_path (str, optional): 목표경로. Defaults to "".
            path_sep (str, optional): 경로구분자. Defaults to "/".

        Returns:
            str: 확인된 폴더경로를 반환합니다.
        """
        dump_dir_path = (
            self._data_root_path.replace("data", "temp")
            if target_path == ""
            else os.path.dirname(target_path)
        )
        parent_dir_list = dump_dir_path.split(path_sep)
        target_dir_path = "/"
        for dir_name in parent_dir_list:
            if target_dir_path == "/":
                target_dir_path += dir_name
                continue
            target_dir_path += path_sep + dir_name
            if os.path.exists(target_dir_path):
                continue
            os.makedirs(target_dir_path)
        return target_dir_path

    # end : protected string dump_dir_path_conf()
    # '''

    @abstractmethod
    def _load_metadata_(self) -> None:
        """
        데이터 셋의 메타데이터를 불러옵니다.
        """
        raise NotImplementedError()

    # protected void load_metadata()

    def _load_decisions_(
        self,
        file_path: str = None,
    ) -> None:
        """
        - 요약:
            - 의사결정 데이터를 불러옵니다.

        - 매개변수:
            - file_path (str, optional): 의사결정 파일경로를 정합니다. None이면 원시데이터 셋 모두를 적재합니다.
        """
        if self._decision_dict == None:
            self._decision_dict = dict()
        self._decision_dict.clear()

        if file_path == None:
            for kwd in ["view", "like", "purchase"]:
                file_path = f"{self._data_root_path}/{kwd}_list.csv"
                self._decision_dict[kwd] = BaseAction.load_collection(file_path)
            # end : for (decisions_type)
        else:
            if not os.path.exists(file_path):
                raise FileNotFoundError()
            kwd = os.path.basename(file_path).split("_")[-2]
            self._decision_dict[kwd] = BaseAction.load_collection(file_path)

    # end : protected void load_decisions()

    def relations_mapping_init(self) -> None:
        """
        - 요약:
            - 의사결정에 따른 사용자, 항목, 태그들의 관계표현을 초기화합니다.
        """
        # 사용자
        for user_id in self.user_dict.keys():
            user: UserEntity = self.user_dict[user_id]
            user.relations_clear()
        # end : for (users)

        # 항목
        for item_id in self.item_dict.keys():
            item: ItemEntity = self.item_dict[item_id]
            item.relations_clear()
        # end : for (items)

        # 태그
        for tag_name in self.tags_dict.keys():
            inst: TagEntity = self.tags_dict[tag_name]
            inst.relations_clear()
        # end : for (tags)

    # end : public void relations_mapping_init()

    def __read_users_list__(self) -> bool:
        _file_path = str.format(
            "{0}/{1}/user_list.csv",
            self._data_root_path,
            DataType.to_str(
                self._data_type,
            ),
        )
        if not os.path.exists(_file_path):
            return False
        with open(_file_path, "rt") as fin:
            fin.readline()
            for line in fin.readlines():
                _r = line.split(",")
                uidx = int(_r[0])
                uid = int(_r[1])
                self.user_dict.update({uid: UserEntity(uidx, uid)})
                self.user_id_to_idx.update({uid: uidx})
                self.user_idx_to_id.update({uidx: uid})
            # end : for (datalines)
            fin.close()
        # end : StreamReader()

    # end : private bool read_users_list()

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

        iidx = tidx = 0
        uidx = len(self.user_dict) if self.__read_users_list__() else 0

        for tag_name in self.tags_dict.keys():
            # tag: TagEntity = self.tags_dict[tag_name]
            if tag_name in self.tag_name_to_idx:
                continue
            self.tag_name_to_idx.update({tag_name: tidx})
            self.tag_idx_to_name.update({tidx: tag_name})
            tidx += 1
        # end : for (tags)

        # item tags reallocation
        for item_id in self.item_dict.keys():
            item: ItemEntity = self.item_dict[item_id]
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
                user: UserEntity = self.user_dict[user_id]
                item: ItemEntity = self.item_dict[item_id]
                user.dict_of_decision_item_ids[kwd].add(item_id)
                item.dict_of_users_decision[kwd].add(user_id)

                for tag_name in item.tags_set:
                    # if not tag_name in self.tags_dict:
                    #    continue
                    tag: TagEntity = self.tags_dict[tag_name]

                    """
                    # CorrelationModel에서 집계 함..
                    if not tag_name in user.tags_decision_freq_dict:
                        user.tags_decision_freq_dict.update({tag_name: 0})
                    user.tags_decision_freq_dict[tag_name] += 1
                    """

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

    # end : private void id_index_mapping()

    @property
    def view_list(self) -> list:
        """The collection of class instance (BaseAction)"""
        return self._decision_dict["view"]

    @property
    def like_list(self) -> list:
        """The collection of class instance (BaseAction)"""
        return self._decision_dict["like"]

    @property
    def purchase_list(self) -> list:
        """The collection of class instance (BaseAction)"""
        return self._decision_dict["purchase"]


# end : class
