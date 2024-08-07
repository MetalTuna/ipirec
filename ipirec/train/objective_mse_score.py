## Custom LIB.
from core import BaseObjectiveScore
from core.model.base_estimator import BaseEstimator


class ObjectiveMSEScore(BaseObjectiveScore):

    def __init__(
        self,
        estimator: BaseEstimator,
    ) -> None:
        super().__init__(estimator)

    def _tags_score_cost_(self, decisions_list: list) -> float:
        return super()._tags_score_cost_(decisions_list)

    def _personalization_cost_(self, decisions_list: list) -> float:
        return super()._personalization_cost_(decisions_list)


# end : class
