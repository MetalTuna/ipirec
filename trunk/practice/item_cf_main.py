import os

import pickle

from core import *
from lc_corr import *
from itemcf import *


class ItemCFMain:
    """
    - 요약:
        - 항목기반 협업필터링
            - 유사도 : 피어슨
                - kNN, KMeans 안씀
            - 예측 : 보정된 가중치 합
    """

    def __init__(
        self,
        data_dir_path: str,
        is_debug_mode: bool = True,
    ) -> None:
        self.data_dir_path = data_dir_path
        self.DEBUG_MODE = is_debug_mode
        self._target_data_name = ""
        self._dataset: BaseDataSet = None
        self._model: BaseDistanceModel = None
        self._estimator: BaseEstimator = None
        self._recommender: BaseRecommender = None
        self._evaluator: BaseEvaluator = None

    # end : init()
    """
    def recommendation(self) -> None:
        self._recommender = ELABasedRecommender(
            estimator=self._estimator,
        )
        self._recommender.prediction()
        self._recommender.top_n_recommendation(
            top_n=5,
        )
        self._recommender.threshold_recommendation(
            score_threshold=0.7,
        )

    # end : public void recommendation()
    """

    def load_dataset(self) -> None:
        """데이타 셋 불러오기"""
        dir_name = self.data_dir_path.replace(
            os.path.dirname(self.data_dir_path) + "/", ""
        )
        dump_file_path = ""
        if dir_name == "colley":
            self._target_data_name = "colley"
            # self._dataset = ColleyDataSet(self.data_dir_path)
            self._dataset = ColleyFilteredDataSet(self.data_dir_path)
        elif dir_name == "ml":
            self._target_data_name = "ml"
            # self._dataset = MovieLensDataSet(self.data_dir_path)
            self._dataset = MovieLensFilteredDataSet(self.data_dir_path)
        else:
            raise NotImplementedError()
        # dataset instance generation

        dump_file_path = (
            f"{self._dataset._dump_dir_path}/{self._target_data_name}_dataset.bin"
        )
        if self.DEBUG_MODE:
            if os.path.exists(dump_file_path):
                with open(file=dump_file_path, mode="rb") as fin:
                    self._dataset = pickle.load(fin)
                    fin.close()
                return

        self._dataset.load_dataset()
        if self.DEBUG_MODE:
            with open(file=dump_file_path, mode="wb") as fout:
                pickle.dump(self._dataset, fout)
                fout.close()

    # end : public void load_dataset()

    def analysis(self) -> None:
        """상관관계 계산"""
        dump_file_path = f"{self._dataset._dump_dir_path}/item_sim_pcc.bin"
        if self.DEBUG_MODE:
            if os.path.exists(dump_file_path):
                with open(file=dump_file_path, mode="rb") as fin:
                    self._model: Pearson = pickle.load(fin)
                    fin.close()
                return
        self._model = Pearson(
            dataset=self._dataset,
        )
        self._model.analysis()
        # self._model.__view_probability__()
        if self.DEBUG_MODE:
            with open(file=dump_file_path, mode="wb") as fout:
                pickle.dump(self._model, fout)
                fout.close()
            # end : StreamWriter()

    # end : public void analysis()

    def estimation(self) -> None:
        self._estimator = AdjustedWeightedSum(model=self._model)

    # end : public void estimation()

    def evaluation(self) -> None:
        self._recommender = ScoreBasedRecommender(estimator=self._estimator)
        """
        self._recommender = ELABasedRecommender(
            estimator=self._estimator,
        )
        """
        file_path = f"{self.data_dir_path}/purchase_list.csv"
        self._evaluator = IRMetricsEvaluator(
            self._recommender,
            file_path,
        )
        self._evaluator.top_n_eval(self.rec_conditions_list)
        file_path = f"{self.data_dir_path}/ws_eval_ibcf_aws.csv"
        self._evaluator.evlautions_summary_df().to_csv(
            path_or_buf=file_path,
        )

    # end : public void evaluation()

    def run(self) -> None:
        ## model params
        self.iterations = 20
        self.dist_num = 2
        self.learning_rate = 0.5
        self.generalization = 0.005
        self.rec_conditions_list = [3, 5, 7, 9]

        ## procedure
        self.load_dataset()
        self.analysis()
        self.estimation()
        self.evaluation()

    # end : public void run()


# end : class


if __name__ == "__main__":
    data_dir_path = f"{os.path.dirname(__file__)}/data/colley"
    # data_dir_path = f"{os.path.dirname(__file__)}/data/ml"

    main = ItemCFMain(data_dir_path=data_dir_path)
    main.run()

# end : main()
