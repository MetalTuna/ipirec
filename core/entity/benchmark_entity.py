from pandas import DataFrame

from ..defines import DecisionType
from ..model import BaseRecommender
from ..eval import BaseEvaluator, CosineItemsTagsFreqAddPenalty, TagsScoreRMSEEvaluator


class BenchmarkEntity:

    def __init__(
        self,
    ) -> None:
        raise NotImplementedError()

    # end : init()

    def benchmark(
        self,
        summary_file_path: str,
        decision_type: DecisionType,
        recommender: BaseRecommender,
        metric: BaseEvaluator,
        top_n_conditions: list = [n for n in range(3, 37, 2)],
    ):
        rec_tags_dist = CosineItemsTagsFreqAddPenalty()
        _kfold: int = recommender._config_info.get(
            "DataSet",
            "kfold_set_no",
            None,
        )
        if _kfold == None:
            raise NotImplementedError()
        _test_set_file_path = recommender._dataset.kfold_file_path(
            _kfold,
            decision_type,
            False,
        )
        recommender.prediction()

        score_eval = TagsScoreRMSEEvaluator(
            recommender,
            _test_set_file_path,
        )
        metric.top_n_eval(top_n_conditions)
        metrics_df: DataFrame = metric.evlautions_summary_df()
        _rec_tag_freq_dist: float = rec_tags_dist.tags_freq_distance(
            _test_set_file_path,
            recommender,
        )
        score_eval.eval()

        self.__append_results__(
            metrics_df,
            _rec_tag_freq_dist,
            score_eval,
        )

    # end : public void benchmark()

    def __append_results__(
        self,
        metrics_df: DataFrame,
    ) -> None:
        pass

    # end : private void append_results()


# end : class
