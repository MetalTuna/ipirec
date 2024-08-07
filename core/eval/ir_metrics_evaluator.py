from pandas import DataFrame

from core import BaseRecommender, UserEntity, BaseAction
from .base_evaluator import BaseEvaluator


class IRMetricsEvaluator(BaseEvaluator):

    def __init__(
        self,
        recommender: BaseRecommender,
        file_path: str,
    ) -> None:
        self.__actual_dict = dict()
        """
        - 요약:
            - 정답지 비교를 위한 dictionary
        - 구성:
            - Key: user_id (int)
            - Value: item_ids (set)
        """
        super().__init__(recommender, file_path)
        self.__tp_list = list()
        self.__fp_list = list()
        self.__fn_list = list()
        self.__tn_list = list()

        self.precision_list = list()
        self.recall_list = list()
        self.f1_score_list = list()
        self.accuracy_list = list()

    # end : init()

    def __load_test_set__(self) -> None:
        """정답지 비교를 위한 dict 생성"""
        super().__load_test_set__()
        # self.__actual_dict.clear()
        for inst in self.TEST_SET_LIST:
            inst: BaseAction
            user_id = inst.user_id
            if not user_id in self.__actual_dict:
                self.__actual_dict.update({user_id: set()})
            items: set = self.__actual_dict[user_id]
            items.add(inst.item_id)
        # end : for (test_sets)

    # end : private void load_test_set()

    def __member_var_init__(self) -> None:
        # elements of confusion matrix
        self.__tp_list.clear()
        self.__fp_list.clear()
        self.__fn_list.clear()
        self.__tn_list.clear()

        # elements of IR evaluation metrics
        self.precision_list.clear()
        self.recall_list.clear()
        self.f1_score_list.clear()
        self.accuracy_list.clear()

    # end : private void member_var_init()

    def eval(self) -> None:
        ## 추천결과에 대한 오차를 구하니까, 추천목록을 모두 반복하며 오차를 누산
        tp, fp, fn, tn, hits = 0, 0, 0, 0, 0

        ## 추천조건에 대한 추천결과의 혼동행렬 계산
        for user_id in self.user_dict.keys():
            user: UserEntity = self.user_dict[user_id]
            predicted_items_set = set(user.recommended_items_dict.keys())
            if not user_id in self.__actual_dict:
                continue
            real_items_set: set = self.__actual_dict[user_id]
            hit_items_set = predicted_items_set.intersection(real_items_set)

            hits_len = len(hit_items_set)
            residual_positives_len = len(predicted_items_set.difference(hit_items_set))
            residual_negatives_len = len(real_items_set.difference(hit_items_set))
            tp += hits_len
            fp += residual_positives_len
            fn += residual_negatives_len
            tn += (
                self._recommender.items_count
                - hits_len
                - residual_positives_len
                - residual_negatives_len
            )
            hits += hits_len
        # end : for (users)

        ## 혼동행렬 원소를 추가
        self.__tp_list.append(tp)
        self.__fp_list.append(fp)
        self.__fn_list.append(fn)
        self.__tn_list.append(tn)
        self.hits_count_list.append(hits)

        # 혼동행렬에 따른 척도 값들 추가
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1_score = (
            0.0
            if (precision + recall) == 0.0
            else (2 * (precision * recall)) / (precision + recall)
        )
        accuracy = (tp + tn) / (tp + fp + fn + tn)

        # round
        _digit = 6
        precision = round(precision, _digit)
        recall = round(recall, _digit)
        f1_score = round(f1_score, _digit)
        accuracy = round(accuracy, _digit)

        # append metrics
        self.precision_list.append(precision)
        self.recall_list.append(recall)
        self.f1_score_list.append(f1_score)
        self.accuracy_list.append(accuracy)

    # end : eval()

    def evlautions_summary_df(self) -> DataFrame:
        return DataFrame(
            {
                "Conditions": self._conditions_list,
                "Precision": self.precision_list,
                "Recall": self.recall_list,
                "F1-score": self.f1_score_list,
                "Accuracy": self.accuracy_list,
                "Hits": self.hits_count_list,
                "TP": self.__tp_list,
                "FP": self.__fp_list,
                "FN": self.__fn_list,
                "TN": self.__tn_list,
            }
        )

    # end : public DataFrame evlautions_summary_df()


# end : class
