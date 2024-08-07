import os
from abc import *
from configparser import ConfigParser
import copy

# from _typeshed import OpenTextMode

from pandas import DataFrame

# from core import BaseRecommender, BaseDataSet, BaseModel, BaseAction
from ..model import BaseRecommender, BaseDataSet, BaseModel
from ..entity import BaseAction
from ..io import DirectoryPathValidator


class BaseEvaluator(metaclass=ABCMeta):
    """
    - 요약:
        - 모델의 성능평가를 위한 추상 클래스입니다.

    >>> evaluator = IRMetricsEvaluator(recommender)
    >>> evaluator.top_n_eval(top_n_conditions=[3, 5, 7])
    >>> results_df = evaluator.evlautions_summary_df()

    ----
    - 멤버함수:
        - private void load_test_set(): 검증용 데이터 셋을 불러옵니다.
        - private void eval_vars_init(): 평가척도 관련변수들을 초기화합니다.
        - public void top_n_eval(top_n_conditions: list): 조건별 추천결과를 구합니다.
            - 조건에 따른 추천 후, eval()를 호출해 추천결과를 평가합니다.
        - public void threshold_eval(threshold_conditions: list): 조건별 추천결과를 구합니다.
            - 조건에 따른 추천 후, eval()를 호출해 추천결과를 평가합니다.

    ----
    - 추상함수:
        - public void eval(): 조건에 대한 추천결과를 구하는 추상함수입니다.
        - public DataFrame evlautions_summary_df(): 조건에 따른 추천결과의 평가결과를 DataFrame으로 출력합니다.
        - private void member_var_init(): 평가척도에 사용될 추가변수들을 초기화합니다.
    """

    def __init__(
        self,
        recommender: BaseRecommender,
        file_path: str,
    ) -> None:
        """
        요약:
            추천모델의 성능평가를 위한 추상클래스입니다.

        매개변수:
            recommender (BaseRecommender): trained recommender instance를 사용해야합니다.
            _test_set_file_path (str): test-set 파일경로 입니다.
        """
        if (not os.path.exists(file_path)) or (os.path.isdir(file_path)):
            raise FileNotFoundError()
        self._test_set_file_path = file_path
        """검증용 데이터 셋의 파일경로"""
        self._recommender = recommender
        """추천모델 객체"""
        self.TEST_SET_LIST: list = None
        """검증데이터 셋 (list)"""
        self.hits_count_list = list()
        """추천조건에 따른 hits결과 목록입니다."""
        self._conditions_list: list = None
        """추천조건 목록입니다."""
        self.__load_test_set__()

    # end : init()

    def save_evaluations_summary(
        self,
        file_path: str = "",
        mode="at",
    ) -> DataFrame:
        """
        - 요약:
            - 평가결과를 지정된 경로에 저장합니다.

        - 매개변수:
            - file_path (str, optional): 결과를 저장할 파일경로입니다. 기본 값은 ""입니다.
            - mode (OpenTextMode, optional): 중복된 파일이 있다면, 결과를 뒤에 붙여씁니다.
                - 기본 값은 AppendText입니다.

        - 반환:
            - DataFrame: 출력된 실험결과에 대한 인스턴스를 반환합니다.
        """

        __dir_path_str = ""
        __dest_file_name_str = ""
        """
        __dir_path_str = (
            os.path.dirname(__file__).replace("core/eval", "results")
            if file_path == ""
            else os.path.dirname(file_path)
        )
        """

        if file_path == "":
            __dir_path_str = os.path.dirname(__file__).replace("core/eval", "results")
            __dest_file_name_str = (
                f"{DirectoryPathValidator.current_datetime_str()}.csv"
            )
        else:
            __dir_path_str = os.path.dirname(file_path)
            __dest_file_name_str = os.path.basename(file_path)
        if not DirectoryPathValidator.exist_dir(__dir_path_str):
            DirectoryPathValidator.mkdir(__dir_path_str)
        file_path = f"{__dir_path_str}/{__dest_file_name_str}"

        df = self.evlautions_summary_df().to_csv(
            path_or_buf=file_path,
            mode=mode,
            encoding="utf-8",
        )

        ## [IO] dump model Vars.
        file_path = str.format(
            "{0}/{1}",
            __dir_path_str,
            __dest_file_name_str.replace("csv", "ini"),
        )
        with open(
            file=file_path,
            mode="wt",
            encoding="utf-8",
        ) as fout:
            self._config.write(fp=fout)
            fout.close()
        # end : StreamWriter()

        return df

    # end : public DataFrame save_evaluations_summary()

    def __eval_vars_init__(self) -> None:
        """평가척도 관련변수들을 초기화합니다."""
        self.hits_count_list.clear()
        self.__member_var_init__()

    # end : private void eval_vars_init()

    def _append_config_(
        self,
        condition_type: str,
        conditions_list: list,
    ) -> None:
        """_summary_

        Args:
            condition_type (str): Threshold OR TopN
            conditions_list (list): conditions_list
        """

        ## cp config_inst >> self
        config: ConfigParser = copy.deepcopy(
            self._recommender._esimator._config_info,
        )

        ## [Recommender]
        _section = "Recommender"
        if config.has_section(_section):
            config.remove_section(_section)
        config.add_section(_section)
        # recommender_name
        _option = "name"
        _value = self._recommender._recommender_name
        config.set(
            section=_section,
            option=_option,
            value=_value,
        )
        # recommender_conditions
        _option = "condition_type"
        _value = condition_type
        config.set(
            section=_section,
            option=_option,
            value=_value,
        )
        _option = "conditions_list"
        _value = ""
        for condition in conditions_list:
            _value += f"{condition},"
        # end : for (conditions)
        _value = _value[0 : len(_value) - 1]
        config.set(
            section=_section,
            option=_option,
            value=_value,
        )

        ## [Evaluator]
        _section = "Evaluator"
        if config.has_section(_section):
            config.remove_section(_section)
        config.add_section(_section)

        # metric_name
        _option = "name"
        _value = self._evaluator_name
        config.set(
            section=_section,
            option=_option,
            value=_value,
        )
        self._config = config

    # end : protected void append_config()

    def __load_test_set__(self) -> None:
        """검증용 데이터 셋을 불러옵니다."""
        loaded_list = list()
        for inst in BaseAction.load_collection(self._test_set_file_path):
            inst: BaseAction
            if not inst.user_id in self.user_dict:
                continue
            if not inst.item_id in self.item_dict:
                continue
            loaded_list.append(inst)
        # end : for (test_sets)
        self.TEST_SET_LIST = loaded_list

    # private void load_test_set()

    @abstractmethod
    def eval(
        self,
    ) -> None:
        """
        - 요약:
            - 성능평가!
        - 유의:
            - 각 조건별로 정답에 속하는 추천항목의 개수를 self.hits_count_list에 추가할 것.
        """
        raise NotImplementedError()

    # end : public void eval()

    @abstractmethod
    def evlautions_summary_df(self) -> DataFrame:
        """
        - 요약:
            - 조건에 따른 추천결과의 평가결과를 DataFrame로 출력합니다.
        - 반환:
            DataFrame: 추천결과를 DataFrame으로 요약합니다.
        """
        raise NotImplementedError()

    # end : public DataFrame evlautions_summary_df()

    @abstractmethod
    def __member_var_init__(self) -> None:
        """추가변수들을 초기화합니다."""
        raise NotImplementedError()

    def threshold_eval(
        self,
        threshold_conditions: list,
    ) -> None:
        """
        - 요약:
            - 예측점수가 임계 값 이상인 항목들을 추천합니다.

        Args:
            threshold_conditions (list): 예측점수의 임계 값에 대한 조건목록(float)
        """
        self.__eval_vars_init__()
        self._append_config_(
            condition_type="Threshold",
            conditions_list=threshold_conditions,
        )
        self._conditions_list = threshold_conditions
        for threshold in threshold_conditions:
            self._recommender.threshold_recommendation(score_threshold=threshold)
            self.eval()
        # end : for (threshold_conditions)

    # end : public void threshold_eval()

    def top_n_eval(
        self,
        top_n_conditions: list,
    ) -> None:
        """
        - 요약:
            - 상위 N개의 예측점수에 속하는 항목들을 추천합니다.
        """
        self.__eval_vars_init__()
        self._append_config_(
            condition_type="TopN",
            conditions_list=top_n_conditions,
        )
        self._conditions_list = top_n_conditions
        for top_n in top_n_conditions:
            self._recommender.top_n_recommendation(top_n)
            self.eval()
        # end : for (top_n_conditions)

    # end : public void top_n_eval()

    ### properites
    @property
    def model(self) -> BaseModel:
        return self._recommender._esimator._model

    ### direct reference propeties
    @property
    def dataset(self) -> BaseDataSet:
        return self._recommender._dataset

    @property
    def user_dict(self) -> dict:
        return self._recommender._dataset.user_dict

    @property
    def item_dict(self) -> dict:
        return self._recommender._dataset.item_dict

    @property
    def tags_dict(self) -> dict:
        return self._recommender._dataset.tags_dict

    @property
    def view_list(self) -> list:
        """학습데이터에서의 왔다목록"""
        return self._recommender._dataset._decision_dict["view"]

    @property
    def like_list(self) -> list:
        """학습데이터에서의 좋다목록"""
        return self._recommender._dataset._decision_dict["like"]

    @property
    def purchse_list(self) -> list:
        """학습데이터에서의 샀다목록"""
        return self._recommender._dataset._decision_dict["purchase"]

    @property
    def user_id_to_idx(self) -> dict:
        return self._recommender._dataset.user_id_to_idx

    @property
    def user_idx_to_id(self) -> dict:
        return self._recommender._dataset.user_idx_to_id

    @property
    def item_id_to_idx(self) -> dict:
        return self._recommender._dataset.item_id_to_idx

    @property
    def item_idx_to_id(self) -> dict:
        return self._recommender._dataset.item_idx_to_id

    @property
    def tag_name_to_idx(self) -> dict:
        return self._recommender._dataset.tag_name_to_idx

    @property
    def _evaluator_name(self) -> str:
        return type(self).__name__


# end : class
