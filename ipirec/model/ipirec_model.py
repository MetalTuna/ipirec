from core import BaseDataSet, UserEntity
from .legacy.correlation_model import CorrelationModel


class IPIRecModel(CorrelationModel):

    def __init__(
        self,
        dataset: BaseDataSet,
        model_params: dict,
    ) -> None:
        super().__init__(
            dataset,
            model_params,
        )

    # end : init()

    def __top_n_decision_tags__(self) -> None:
        super().__top_n_decision_tags__()
        for user_id in self.user_dict.keys():
            user: UserEntity = self.user_dict[user_id]
            _appended = set.union(
                user.top_n_decision_tags_set, user.set_of_interest_tags
            )
            user.top_n_decision_tags_set = _appended
        # end : for (users)


# end : class
