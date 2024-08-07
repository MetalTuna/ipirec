import os
import sys

import numpy as np
from pandas import DataFrame

__dir_path_str__ = os.path.dirname(__file__)
__dir_name__ = os.path.basename(__dir_path_str__)
WORKSPACE_HOME = __dir_path_str__.replace(f"/{__dir_name__}", "")
sys.path.append(WORKSPACE_HOME)

from core import *
from lc_corr import *
from decompositions import *
from itemcf import *


class Validator:
    """
    사용금지
    =====
    사유: 구현 중;
    """

    def __init__(
        self,
        conditions_dict: dict,
    ) -> None:
        self._WORKSPACE_PATH = DirectoryPathValidator.get_workspace_path("ipirec")
        """workspace_home_path_str"""
        self._RAW_DATASET_HOME_PATH = f"{self._WORKSPACE_PATH}/data"
        """raw dataset home path str"""
        self._DATASET_TYPE: DataType = None
        """dataset (type) = COLLEY or MOVIELENS"""

        self._conditions_dict = conditions_dict
        """
        Key: RecommenderOption [ E_TOP_N, E_SCORE_THRESHOLD ]
        Value: conditions_list
        """
        raise NotImplementedError()

    # end : public void init()

    def run(
        self,
        data_type: DataType,
        model_type: AnalysisMethodType,
        recommender_type: RecommenderType,
        metric_type: MetricType,
    ) -> None:
        self.raw_dataset_dir = f"{self._RAW_DATASET_HOME_PATH}/"
        dataset: BaseDataSet = None
        self.raw_dataset_dir += DataType.to_str(data_type)
        match (data_type):
            case DataType.E_COLLEY:
                dataset = ColleyFilteredDataSet(self.raw_dataset_dir)
            case DataType.E_MOVIELENS:
                dataset = MovieLensFilteredDataSet(self.raw_dataset_dir)
            case _:
                raise ValueError()
        # end : match-case
        self._DATA_TYPE = data_type
        dataset._load_metadata_()

        # 교차검증 데이터 셋 구성
        splitter = CrossValidationSplitter(self.raw_dataset_dir)
        splitter.split()
        # 교차검증에 대한 분석 및 결과평가
        for kfold in splitter.validations_decisions_dict.keys():
            validaiton_dict: dict = splitter.validations_decisions_dict[kfold]
            train_list: list = validaiton_dict["train"]
            test_list: list = validaiton_dict["test"]
            # file_name = [train, test]_[0, ..., k - 1]_[view, like, purchase]_list,
            # => train_0_view_list.csv

            ## init member vars.
            dataset.relations_mapping_init()
            for decision_type in DecisionType:
                kwd = DecisionType.to_str(decision_type)
                file_path = str.format(
                    "{0}/train_{1}_{2}_list.csv",
                    self.raw_dataset_dir,
                    kfold,
                    kwd,
                )
                dataset.append_decisions(
                    file_path=file_path,
                    decision_type=decision_type,
                )
            # end : for (decision_type)
            dataset.__id_index_mapping__()

            model: BaseModel = None
            estimator: BaseEstimator = None
            recommender: BaseRecommender = None
            evaluator: BaseEvaluator = None

            match (model_type):
                case AnalysisMethodType.E_IBCF:
                    model = Pearson(
                        dataset=data_type,
                    )
                    estimator = AdjustedWeightedSum(model=model)
                case AnalysisMethodType.E_IPIRec:
                    model = CorrelationModel(
                        dataset=dataset,
                    )
                    estimator = BiasedEstimator(model=model)
                case AnalysisMethodType.E_NMF:
                    model = DecompositionModel(
                        dataset=dataset,
                    )
                    estimator = DecompositionsEstimator(model=model)
                case _:
                    raise NotImplementedError()
            # end : match-case (MODEL)

            for decision_type in DecisionType:
                estimator.train(decision_type)
            # end : for (decisions)

            # [DEF] Recommender
            match (recommender_type):
                case RecommenderType.E_ELA:
                    recommender = ELABasedRecommender(estimator=estimator)
                case RecommenderType.E_SCORE:
                    recommender = ScoreBasedRecommender(estimator=estimator)
                case _:
                    raise NotImplementedError()
            # end : match-case (REC.)

            match (metric_type):
                case MetricType.E_RETRIEVAL:
                    evaluator = IRMetricsEvaluator(recommender=recommender)
                case MetricType.E_STATISTICS:
                    evaluator = StatisticsEvaluator(recommender=recommender)
                case _:
                    raise NotImplementedError()
            # end : match-case (EVAL.)

            # [Opt.] Recommender
            recommender.prediction()
            for rec_opt in RecommenderOption:
                conditions_list = self._conditions_dict[rec_opt]
                opt_key: str = RecommenderOption.to_str(rec_opt)
                condition_df = DataFrame({opt_key: dict()})
                match (rec_opt):
                    case RecommenderOption.E_TOP_N:
                        for top_n in conditions_list:
                            recommender.top_n_recommendation(top_n=top_n)
                            evaluator.eval()
                            condition_df.insert(
                                opt_key,
                                top_n,
                                evaluator.evlautions_summary_df(),
                            )
                        # end : for (top_n_conditions)
                    case RecommenderOption.E_SCORE_THRESHOLD:
                        for score_threshold in conditions_list:
                            recommender.threshold_recommendation(
                                score_threshold=score_threshold
                            )
                            evaluator.eval()
                            condition_df.insert(
                                opt_key,
                                score_threshold,
                                evaluator.evlautions_summary_df(),
                            )
                        # end : for (threshold_conditions)
                    case _:
                        raise NotImplementedError()
                # en : match-case (RecOpt.)
            # end : for (RecOpt.)
        # end : for (k-fold CV)

        # 결과 집계/저장/요약

    # end : public void run()


# end : class

if __name__ == "__main__":
    experiments = Validator()
    K_FOLD = 5

    top_n_list = [n for n in range(3, 37, 2)]
    threshold_list = [th for th in np.arange(0.1, 1.0, 0.1)]
    conditions_dict = {
        RecommenderOption.E_TOP_N: top_n_list,
        RecommenderOption.E_SCORE_THRESHOLD: threshold_list,
    }

    selected_metrics = MetricType.E_RETRIEVAL
    for selected_data in DataType:
        for selected_model in AnalysisMethodType:
            for selected_recommender in RecommenderType:
                for selected_rec_option in RecommenderOption:
                    experiments.run(
                        data_type=selected_data,
                        model_type=selected_model,
                        recommender_type=selected_recommender,
                        metric_type=selected_metrics,
                    )
                    """
                    for selected_metrics in MetricType:
                        experiments.run(
                            data_type=selected_data,
                            model_type=selected_model,
                            recommender_type=selected_recommender,
                            metric_type=selected_metrics,
                        )
                        # eval.run()
                    # end : for (Metrics)
                    """
                # end : for (RecommenderOption)
            # end : for (Recommender)
        # end : for (Model)
    # end : for (DataType)
# end : main()
