### buildt-in
import os
import sys
from abc import *

### 3rd Pty.
import numpy as np

### Custom LIB
__CURRRENT_DIR_PATH = os.path.dirname(__file__)
WORKSPACE_HOME = __CURRRENT_DIR_PATH.replace(
    f"/experiments/{os.path.basename(__CURRRENT_DIR_PATH)}", ""
)
sys.path.append(WORKSPACE_HOME)
# print(WORKSPACE_HOME)
# exit()

from core import *
from lc_corr import *
from rec_tags_freq import *


class BaseTagsDistribution(metaclass=ABCMeta):
    def __init__(
        self,
        dataset_home_dir: str,
        model_dump_path: str,
        figures_dump_path: str,
    ) -> None:
        self._DATASET_DIR = ""
        self._MODEL_DUMP_DIR = ""
        self._FIGURES_DUMP_DIR = ""

        self._DATASET_TYPE: DataType = None
        self._dataset: BaseDataSet = None
        self._model: BaseModel = None
        self._estimator: BaseEstimator = None
        self._recommender: BaseRecommender = None
        self._selected_decision: DecisionType = None
        self._testset_list: list = None
        self._validation_set_id: int = int()

        self._dir_path_validation_(
            dataset_home_dir,
            model_dump_path,
            figures_dump_path,
        )

    # end : init()

    @property
    def user_dict(self) -> dict:
        return self._dataset.user_dict

    @property
    def item_dict(self) -> dict:
        return self._dataset.item_dict

    @property
    def tags_dict(self) -> dict:
        return self._dataset.tags_dict

    def _dir_path_validation_(
        self,
        dataset_home_dir: str,
        model_dump_path: str,
        figures_dump_path: str,
    ) -> None:
        ### [Validation] directories path;
        # DataSet
        dir_path = os.path.dirname(dataset_home_dir)
        if not DirectoryPathValidator.exist_dir(dir_path):
            raise FileNotFoundError()
        self._DATASET_DIR = dir_path
        self._DATASET_TYPE = DataType.dir_path_str_to_inst(self._DATASET_DIR)

        # Model
        dir_path = os.path.dirname(model_dump_path)
        if not DirectoryPathValidator.exist_dir(dir_path):
            DirectoryPathValidator.mkdir(dir_path)
        self._MODEL_DUMP_DIR = dir_path

        # Figures
        dir_path = os.path.dirname(figures_dump_path)
        if not DirectoryPathValidator.exist_dir(dir_path):
            DirectoryPathValidator.mkdir(dir_path)
        self._FIGURES_DUMP_DIR = dir_path

    # end : protected void dir_path_validation()

    def top_n_recommended_tags_distance(self, top_n_conditions: list) -> None:
        raise NotImplementedError()

    # end : public void top_n_recommended_tags_distance()

    def tags_distance(self) -> None:
        self._load_testset_()
        tags_freq_dist = CosineItemsTagsFreqAddPenalty()
        distance: float = tags_freq_dist.tags_freq_distance(
            test_set=self._testset_list,
            recommender=self._recommender,
        )

        print(f"tags freq. distance (cosine): {distance}")

    # end : public void tags_distribution()

    def _load_testset_(self) -> None:
        file_path = str.format(
            "{0}/test_{1}_{2}_list.csv",
            self._DATASET_DIR,
            self._validation_set_id,
            DecisionType.to_str(self._selected_decision),
        )

        self._testset_list: list = [
            inst
            for inst in BaseAction.load_collection(file_path)
            if ((inst.user_id in self.user_dict) and (inst.item_id in self.item_dict))
        ]

    # end : protected void load_testset()

    @abstractmethod
    def _build_model_(
        self,
        validation_set_id: int,
        selected_decision: DecisionType,
    ):
        raise NotImplementedError()

    @abstractmethod
    def _extern_distribution_(self):
        raise NotImplementedError()

    def __test__(self):
        file_name = str.format(
            "{0}_{1}",
            self._model.model_name,
            DataType.to_str(self._DATASET_TYPE),
        )


# end : class


if __name__ == "__main__":

    dataset_dir_path = f"{WORKSPACE_HOME}/data/colley"
    testset_file_path = f"{dataset_dir_path}/like_list.csv"
    # testset_file_path = f"{dataset_dir_path}/purchase_list.csv"
    results_summary_path = f"{WORKSPACE_HOME}/results/ipa_colley_TopN_IRMetrics.csv"
    raw_corr_figure_file_path = f"{WORKSPACE_HOME}/results/ipa_colley_raw_tags_corr.svg"
    biased_corr_figure_file_path = (
        f"{WORKSPACE_HOME}/results/ipa_colley_biased_tags_corr.svg"
    )
    corr_weight_figure_file_path = (
        f"{WORKSPACE_HOME}/results/ipa_colley_tags_corr_wegiht.svg"
    )
    adjusted_corr_figure_file_path = (
        f"{WORKSPACE_HOME}/results/ipa_colley_adjuted_tags_corr.svg"
    )

    # 모델의 매개변수들
    dist_n = 1
    learning_rate = 0.01
    generalization = 0.5
    co_occur_items_threshold = 20
    iterations_threshold = 100
    adjust_iterations = 20
    top_n_tags = 10
    top_n_conditions = [n for n in range(3, 21, 2)]

    # 데이터 셋 불러오기
    dataset = ColleyFilteredDataSet(dataset_dir_path=dataset_dir_path)
    dataset.load_dataset()

    # 모델 구성하기

    model_params = CorrelationModel.create_models_parameters(
        top_n_tags=top_n_tags,
        co_occur_items_threshold=co_occur_items_threshold,
        iterations_threshold=iterations_threshold,
        learning_rate=learning_rate,
    )
    model = CorrelationModel(
        dataset=dataset,
        model_params=model_params,
    )
    model.analysis()

    # [DRAW] heatmap plot
    HeatmapFigure.draw_heatmap(
        tag_idx_to_name_dict=model.tag_idx_to_name,
        tags_corr=model.arr_tags_score,
        fig_title="raw_tags_score",
        file_path=raw_corr_figure_file_path,
    )

    # 학습하기
    model_params = BiasedEstimator.create_models_parameters(
        learning_rate=learning_rate,
        generalization=generalization,
    )
    estimator = BiasedEstimator(
        model=model,
        model_params=model_params,
    )

    # [DRAW] heatmap plot
    HeatmapFigure.draw_heatmap(
        tag_idx_to_name_dict=model.tag_idx_to_name,
        tags_corr=estimator.model.arr_tags_score,
        fig_title="biased_tags_score",
        file_path=biased_corr_figure_file_path,
    )

    estimator.train(
        DecisionType.E_VIEW,
        n=dist_n,
        emit_iter_condition=adjust_iterations,
    )
    estimator.train(
        DecisionType.E_LIKE,
        n=dist_n,
        emit_iter_condition=adjust_iterations,
    )
    estimator.train(
        DecisionType.E_PURCHASE,
        n=dist_n,
        emit_iter_condition=adjust_iterations,
    )

    # [DRAW] heatmap plot
    tags_weight: np.ndarray = estimator.arr_user_idx_to_weights.mean(axis=0)
    HeatmapFigure.draw_heatmap(
        tag_idx_to_name_dict=model.tag_idx_to_name,
        tags_corr=tags_weight,
        fig_title="tags_score_weight",
        file_path=corr_weight_figure_file_path,
        diag_score=1.0,
    )
    weighted_tags_corr = np.multiply(
        estimator.arr_tags_score, tags_weight
    )  # estimator.arr_tags_score * tags_weight
    HeatmapFigure.draw_heatmap(
        tag_idx_to_name_dict=model.tag_idx_to_name,
        tags_corr=weighted_tags_corr,
        fig_title="adjusted_tags_score",
        file_path=adjusted_corr_figure_file_path,
    )

    # 예측 점수를 기준으로 추천하기
    recommender = ScoreBasedRecommender(estimator=estimator)
    recommender.prediction()

    # 성능평가하기
    evaluator = IRMetricsEvaluator(
        recommender=recommender,
        file_path=testset_file_path,
    )
    evaluator.top_n_eval(top_n_conditions=top_n_conditions)
    evaluator.evlautions_summary_df().to_csv(path_or_buf=results_summary_path)

    # tags_freq_dist = CosineItemsTagsFreq()
    tags_freq_dist = CosineItemsTagsFreqAddPenalty()
    distance: float = tags_freq_dist.tags_freq_distance(
        test_set=evaluator.TEST_SET_LIST,
        recommender=recommender,
    )

    print(f"tags freq. distance (cosine): {distance}")
# end : main()
