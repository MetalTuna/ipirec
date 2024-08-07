import os

import pickle

from core import *
from lc_corr import *
from decompositions import *


class DecompositionsMain:
    """
    - 요약:
        - 행렬분해모델
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
        self._model: BaseModel = None
        self._estimator: BaseEstimator = None
        self._recommender: BaseRecommender = None
        self._evaluator: BaseEvaluator = None

    # end : init()

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
            f"{self._dataset._dump_dir_path}/{self._target_data_name}_nmf_dataset.bin"
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
        # """
        self._model = NMFDecompositionModel(
            dataset=self._dataset,
            factors_dim=self.factors_dim,
        )
        # """
        """
        self._model = TruncatedSVDModel(
            dataset=self._dataset,
            factors_dim=self.factors_dim,
            factorizer_iters=self.iterations,
        )
        """
        self._model.analysis()

    # end : public void analysis()

    def estimation(self) -> None:
        self._estimator = DecompositionsEstimator(
            model=self._model,
            # factors_dim=self.factors_dim,
            learning_rate=self.learning_rate,
            generalization=self.generalization,
        )

        """적합한 잠재특성 찾기"""
        self._estimator.train(
            target_decision=DecisionType.E_VIEW,
            n=self.dist_num,
            emit_iter_condition=self.iterations,
        )
        self._estimator.train(
            target_decision=DecisionType.E_LIKE,
            n=self.dist_num,
            emit_iter_condition=self.iterations,
        )
        self._estimator.train(
            target_decision=DecisionType.E_PURCHASE,
            n=self.dist_num,
            emit_iter_condition=self.iterations,
        )

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
        file_path = f"{self.data_dir_path}/eval_nmf.csv"
        self._evaluator.evlautions_summary_df().to_csv(
            path_or_buf=file_path,
        )

    # end : public void evaluation()

    def run(self) -> None:
        ## model params
        self.factors_dim = 80
        self.iterations = 20
        self.dist_num = 1
        self.learning_rate = 0.005
        self.generalization = 0.05
        self.rec_conditions_list = [3, 5, 7, 9]

        ## procedure
        self.load_dataset()
        self.analysis()
        self.estimation()
        self.evaluation()

    # end : public void run()


# end : class


if __name__ == "__main__":
    # data_dir_path = f"{os.path.dirname(__file__)}/data/colley"
    data_dir_path = f"{os.path.dirname(__file__)}/data/ml"

    main = DecompositionsMain(data_dir_path=data_dir_path)
    main.run()

# end : main()
