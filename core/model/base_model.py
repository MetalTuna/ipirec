from abc import *
from configparser import ConfigParser

from .base_model_params import BaseModelParameters
from .base_dataset import BaseDataSet
from ..io import InstanceIO


# class BaseModel(metaclass=ABCMeta):
class BaseModel(BaseModelParameters, InstanceIO):
    """
    - 요약:
        - 데이터 분석을 위한 추상클래스입니다.
        - 데이터 입력부터 예측 이전까지의 기능들을 담당하도록 구현합니다.
    - 추상함수:
        - private void preprocess()
        - private void process()
        - private void postprocess()
    """

    ## 모듈 OR 클래스 이름 가져오는 함수 ;
    def __init__(
        self,
        dataset: BaseDataSet,
        model_params: dict = None,
        # is_debug_mode: bool = True,
    ) -> None:
        self._dataset = dataset
        # self.IS_DEBUG_MODE = is_debug_mode

        if model_params != None:
            self._set_model_params_(model_params=model_params)
        self._append_config_(
            model_params=model_params,
            section="Model",
        )

        self._dump_dir_path = ""
        if self.IS_DEBUG_MODE:
            self._dump_dir_path = self._dataset._dump_dir_path_conf_()

    # end : init()

    def _append_config_(
        self,
        model_params: dict,
        section: str = "Model",
        inst=None,
    ) -> None:
        """
        - 요약:
            - ConfigParser에 현재 설정된 모델변수들을 추가합니다.
        """

        ## config
        self._section = section
        if not self._config_info.has_section(self._section):
            self._config_info.add_section(self._section)
        _option = "name"
        _value = self.model_name if inst == None else type(inst).__name__
        self._config_info.set(
            section=self._section,
            option=_option,
            value=_value,
        )
        if model_params != None:
            for key, value in model_params.items():
                _option = str(key)
                _value = str(value)
                self._config_info.set(
                    section=self._section,
                    option=_option,
                    value=_value,
                )
            # end : for (model_params)

    # end : protected void append_config()

    @property
    def IS_DEBUG_MODE(self) -> bool:
        return self._dataset.IS_DEBUG_MODE

    ### member functions
    def analysis(self) -> None:
        self._preprocess_()
        self._process_()
        self._postprocess_()

    # abstract methods
    @abstractmethod
    def _preprocess_(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _process_(self) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _postprocess_(self) -> None:
        raise NotImplementedError()

    ### properties
    @property
    def user_dict(self) -> dict:
        """
        Key: user_id (int)
        Value: instance (Any) -- NOT DEFINED
        """
        return self._dataset.user_dict

    @property
    def item_dict(self) -> dict:
        """
        Key: item_id (int)
        Value: instance (ItemEntity)
        """
        return self._dataset.item_dict

    @property
    def tags_dict(self) -> dict:
        """
        Key: item_id (int)
        Value: instance (TagEntity)
        """
        return self._dataset.tags_dict

    @property
    def user_id_to_idx(self) -> dict:
        return self._dataset.user_id_to_idx

    @property
    def view_list(self) -> list:
        """The collection of class instance (BaseAction)"""
        return self._dataset.view_list

    @property
    def like_list(self) -> list:
        """The collection of class instance (BaseAction)"""
        return self._dataset.like_list

    @property
    def purchase_list(self) -> list:
        """The collection of class instance (BaseAction)"""
        return self._dataset.purchase_list

    @property
    def user_idx_to_id(self) -> dict:
        return self._dataset.user_idx_to_id

    @property
    def item_id_to_idx(self) -> dict:
        return self._dataset.item_id_to_idx

    @property
    def item_idx_to_id(self) -> dict:
        return self._dataset.item_idx_to_id

    @property
    def tag_name_to_idx(self) -> dict:
        return self._dataset.tag_name_to_idx

    @property
    def tag_idx_to_name(self) -> dict:
        return self._dataset.tag_idx_to_name

    @property
    def users_count(self) -> int:
        return self._dataset.users_count

    @property
    def items_count(self) -> int:
        return self._dataset.items_count

    @property
    def tags_count(self) -> int:
        return self._dataset.tags_count

    @property
    def _config_info(self) -> ConfigParser:
        return self._dataset._config_info

    @property
    def model_name(self) -> str:
        return type(self).__name__


# end : class
