## build in
from abc import *
import os
import sys

# [DEF] Environment variables
__dir_name__ = os.path.basename(os.path.dirname(__file__))
WORKSPACE_HOME = os.path.dirname(__file__).replace(
    f"/runners/dataset_builders/{__dir_name__}", ""
)
""".../ipirec"""
DATA_SET_HOME = f"{WORKSPACE_HOME}/data"
""">>> `${WORKSPACE_HOME}`/data"""
MODEL_NAME = "IPIRec"
RESULTS_SUMMARY_HOME = f"{WORKSPACE_HOME}/results/{MODEL_NAME}"
""">>> `${WORKSPACE_HOME}`/results/${MODEL_NAME}"""

# [SET] Env.append($WORKSPACE_HOME)
sys.path.append(WORKSPACE_HOME)

## Custom LIB.
from core import *
from colley import *
from ipirec import *
from movielens import *


class BaseRecommendRunner(metaclass=ABCMeta):
    def __init__(
        self,
        temp_dir_path: str = "",
    ) -> None:
        self.dataset: BaseDataSet = None
        self.model: BaseModel = None
        self.estimator: BaseEstimator = None
        self.recommender: BaseRecommender = None
        self.evaluator: BaseEvaluator = None

        self.__TEMP_DIR_PATH = (
            temp_dir_path
            if temp_dir_path != ""
            else str.format(
                "{0}/model_bins",
                os.path.dirname(__file__).replace(
                    "runner",
                    "temp",
                ),
            )
        )
        if not DirectoryPathValidator.exist_dir(
            dir_path=self.__TEMP_DIR_PATH,
        ):
            DirectoryPathValidator.mkdir(
                dir_path=self.__TEMP_DIR_PATH,
            )

    # end : init()

    @abstractmethod
    def runner(self, **kwargs) -> None:
        raise NotImplementedError()

    @abstractmethod
    def _build_model_(self) -> None:
        raise NotImplementedError()

    # end : protected void build_model()

    @abstractmethod
    def _build_estimator_(self) -> None:
        raise NotImplementedError()
        # end : for (decision_types)

    # end : protected void build_estimator()

    def _load_dataset_(
        self,
        selected_dataset: DataType,
        set_no: int,
        dataset_dir_path: str = "",
    ) -> None:
        """
        - 요약:
            - 데이터 셋을 불러옵니다.

        - 매개변수:
            - selected_dataset (DataType): 분석 데이터 셋 종류 지정
            - set_no (int): 교차검증 데이터 셋 번호 지정
            - dataset_dir_path (str, optional): 분섯 데이터 셋 폴더 경로

        - 예외:
            - FileNotFoundError: 데이터 셋 폴더가 유효하지 않거나, 데이터 셋 누락시 발생
            - ValueError: 분석 데이터 셋 열거자가 부적합하면 발생
            - FileNotFoundError: 교차검증 파일이 없을 때 발생
        """

        ## [IO] Dataset
        # path validation
        dataset_dir_path = (
            dataset_dir_path
            if dataset_dir_path != ""
            else str.format("{0}/{1}", DATA_SET_HOME, DataType.to_str(selected_dataset))
        )
        if not DirectoryPathValidator.exist_dir(dir_path=dataset_dir_path):
            raise FileNotFoundError()
        self.__DATASET_HOME_PATH = dataset_dir_path
        self.__KFOLD_SET_NO = set_no

        # read data
        match (selected_dataset):
            case DataType.E_COLLEY:
                self.dataset = ColleyFilteredDataSet(
                    dataset_dir_path=self.__DATASET_HOME_PATH,
                )
            case DataType.E_MOVIELENS:
                self.dataset = MovieLensFilteredDataSet(
                    dataset_dir_path=self.__DATASET_HOME_PATH,
                )
            case _:
                raise ValueError()
        # end : match-case (DataSet)
        self.dataset._load_metadata_()
        for decision_type in DecisionType:
            file_path = str.format(
                "{0}/train_{1}_{2}_list.csv",
                self.__DATASET_HOME_PATH,
                self.__KFOLD_SET_NO,
                DecisionType.to_str(decision_type),
            )
            if not os.path.exists(file_path):
                raise FileNotFoundError()
            self.dataset.append_decisions(
                file_path=file_path,
                decision_type=decision_type,
            )
        # end : for (decision_types)
        self.dataset.__id_index_mapping__()

    # end : protected void load_dataset()

    def _build_recommender_(
        self,
        selected_recommender: RecommenderType,
    ) -> None:
        match (selected_recommender):
            case RecommenderType.E_ELA:
                self.recommender = ELABasedRecommender(
                    estimator=self.estimator,
                )
            case RecommenderType.E_SCORE:
                self.recommender = ScoreBasedRecommender(
                    estimator=self.estimator,
                )
            case _:
                raise ValueError()
        # end : match-case (recommender)

        self.recommender.prediction()

    # end : protected void build_recommender()

    def _build_evaluator_(
        self,
        selected_evaluator: MetricType,
        target_decision_type: DecisionType,
    ) -> None:
        test_set_file_path = str.format(
            "{0}/train_{1}_{2}_list.csv",
            self.__DATASET_HOME_PATH,
            self.__KFOLD_SET_NO,
            DecisionType.to_str(target_decision_type),
        )
        if not os.path.exists(test_set_file_path):
            raise FileNotFoundError()
        self._test_set_file_path = test_set_file_path

        match (selected_evaluator):
            case MetricType.E_RETRIEVAL:
                self.evaluator = IRMetricsEvaluator(
                    recommender=self.recommender,
                    file_path=test_set_file_path,
                )
            case MetricType.E_STATISTICS:
                self.evaluator = StatisticsEvaluator(
                    recommender=self.recommender,
                    file_path=test_set_file_path,
                )
            case _:
                raise ValueError()
        # end : match-case (Evaluator)

    # end : protected void build_evaluator()


# end : class
