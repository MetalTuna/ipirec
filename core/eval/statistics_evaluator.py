import math

from pandas import DataFrame

from core import BaseRecommender, BaseAction, UserEntity
from .base_evaluator import BaseEvaluator


class StatisticsEvaluator(BaseEvaluator):

    def __init__(
        self,
        recommender: BaseRecommender,
        file_path: str,
    ) -> None:
        super().__init__(recommender, file_path)

        self.__mae_list = list()
        self.__rmse_list = list()

    # end : init()

    def __member_var_init__(self) -> None:
        self.__mae_list.clear()
        self.__rmse_list.clear()

    # end : private void member_var_init()

    def eval(self) -> None:
        ## 검증데이터에 있는 추천결과에 대해서만 오차를 구하니까, 검증데이터로 오차를 누산
        mae_numer, rmse_numer, hits = 0.0, 0.0, 0

        for inst in self.TEST_SET_LIST:
            inst: BaseAction
            user: UserEntity = self.user_dict[inst.user_id]
            if not inst.item_id in user.recommended_items_dict:
                continue
            error_rate = 1 - user.recommended_items_dict[inst.item_id]
            mae_numer += math.fabs(error_rate)
            rmse_numer += math.pow(error_rate, 2.0)
            hits += 1
        # end : for (test_set)

        if hits == 0:
            raise NotImplementedError()
        mae_numer = mae_numer / hits
        rmse_numer = math.sqrt(rmse_numer / hits)
        self.__mae_list.append(mae_numer)
        self.__rmse_list.append(rmse_numer)
        self.hits_count_list.append(hits)

    # end : public void eval()

    def evlautions_summary_df(self) -> DataFrame:
        return DataFrame(
            {
                "Conditions": self._conditions_list,
                "MAE": self.__mae_list,
                "RMSE": self.__rmse_list,
                "Hits": self.hits_count_list,
            }
        )

    # end : public dict evlautions_summary_df()


# end : class
