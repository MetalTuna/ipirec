import math

from tqdm import tqdm
import numpy as np

from core import (
    BaseAction,
    BaseModel,
    BaseTrain,
    UserEntity,
    ItemEntity,
    TagEntity,
    DecisionType,
)
from .correlation_model import CorrelationModel


class CorrelationEstimator(BaseTrain):
    """
    - 요약:
        - tags_score가 0인 것들을 \\mu(t)로 채웁니다. (목적)
        - 문서내용과 같이, 사용자와 항목에 따른 태그 유사도로 점수를 구하고 있음 (현황)
            - tags_score를 어떻게 쓸 것인가 ??
    >>> Pr, Corr, score가 0인 값을 bias_score로 초기화하도록 구성합시다.

    Args:
        BaseTrain (_type_): _description_
    """

    def __init__(
        self,
        model: BaseModel,
        model_params: dict,
        # learning_rate: float = 0.1,
        # generalization: float = 0.5,
    ) -> None:
        super().__init__(
            model=model,
            model_params=model_params,
        )

        ### member variables
        self._arr_users_bias = np.zeros(shape=(self.users_count, 1))
        self._arr_items_bias = np.zeros(shape=(self.items_count, 1))
        self._arr_tags_bias = np.zeros(shape=(self.tags_count, 1))
        self._DEFAULT_VOTING = float()
        """
        - 요약:
            - NT(u)에 target tag가 미포함일 때, 반환되는 값입니다.
            - 0~1사이의 값을 갖습니다. (절대 값)
        """

        self._preprocess_()

    # end : init()

    def _tags_score_computation_(self) -> None:
        """태그점수를 계산합니다."""
        raise NotImplementedError()

    # end : protected void tags_score_computation()

    def _adjust_tags_score_(self) -> None:
        """태그점수를 보정합니다."""
        raise NotImplementedError()

    # end : protected void adjust_tags_score()

    @staticmethod
    def create_models_parameters(
        default_voting_score: float = 0.0,
        learning_rate: float = 0.1,
        generalization: float = 0.5,
    ) -> dict:

        ## model params creation;
        model_params = BaseTrain.create_models_parameters(
            learning_rate=learning_rate,
            generalization=generalization,
        )

        ## values validation;
        default_voting_score = (
            default_voting_score if default_voting_score <= 1.0 else 1.0
        )
        default_voting_score = (
            default_voting_score if default_voting_score >= 0.0 else 0.0
        )

        ## append parameter;
        model_params.update({"default_voting_score": default_voting_score})
        return model_params

    def _set_model_params_(
        self,
        model_params: dict,
    ) -> None:
        kwd = "default_voting_score"
        if not kwd in model_params:
            self._DEFAULT_VOTING = 0.0
        self._DEFAULT_VOTING = float(model_params[kwd])
        super()._set_model_params_(model_params)

    # end : protected override void set_model_params()

    ### properties
    @property
    def model(self) -> CorrelationModel:
        return self._model

    @property
    def co_occur_ratio(self) -> np.ndarray:
        """
        co_occur_ratio[x][y] = Pr(x,y) = |I(x) \\cap I(y)| / |I(x)|
        >>> `TagsCoOccurrence[T, T]`
        """
        return self.model.co_occur_ratio

    @property
    def arr_tags_score(self) -> np.ndarray:
        """>>> `TagsScore[T, T]`"""
        return self.model.arr_tags_score

    """
    @arr_tags_score.setter
    def arr_tags_score(self, value) -> None:
        self.model.arr_tags_score = value
    """

    @property
    def _user_based_pcc_tags_array(self) -> np.ndarray:
        """>>> `UsersPCC[T, T]`"""
        return self.model._ub_pcc_co_occur_score

    @property
    def _item_based_pcc_tags_array(self) -> np.ndarray:
        """>>> `ItemsPCC[T, T]`"""
        return self.model._ib_pcc_co_occur_score

    def __fit__(
        self,
        target_decision: DecisionType,
        n: int,
    ) -> float:
        """
        - 요약:
            - 학습데이터 단위로 의사결정 종류에 대한 추정 오차들을 집계합니다.

        - 매개변수:
            - target_decision (DecisionType): 추정할 의사결정 타입을 선택합니다.
            - n (int, optional): Frobenious norm의 n입니다. (기본 값은 1)
                - 유클리드 거리에 근거한 RMSE를 구하려면 n = 2로 하세요.

        - 예외:
            - ValueError: 정의되지 않은 DecisionType을 사용할 경우, 예외가 발생합니다.

        - 반환:
            - float: 오차의 비율을 반환합니다(현재 구조는 거리누산 값의 평균을 구합니다).
        """
        error_rate = 0.0
        decisions_list: list = None
        self.estimation_error_numer = 0.0
        self.estimation_error_denom = 0.0

        tqdm_desc_str = ""
        match (target_decision):
            case DecisionType.E_VIEW:
                decisions_list = self.view_list
                tqdm_desc_str += "[VIEW]"
            case DecisionType.E_LIKE:
                decisions_list = self.like_list
                tqdm_desc_str += "[LIKE]"
            case DecisionType.E_PURCHASE:
                decisions_list = self.purchase_list
                tqdm_desc_str += "[PURCHASE]"
            case _:
                raise ValueError()
        # return self._estimator.estimation(decisions_list)

        # for iter in decisions_list:
        for iter in tqdm(
            iterable=decisions_list,
            desc=f"{tqdm_desc_str} Adjust",
            total=len(decisions_list),
            # leave=False,
        ):
            iter: BaseAction
            if not iter.user_id in self.user_id_to_idx:
                continue
            if not iter.item_id in self.item_id_to_idx:
                continue
            iter = self._estimate_(iter)
            # if iter.estimated_score == 0.0 or iter.estimated_score == 1.0:
            if iter.estimated_score == 1.0:
                continue

            error_rate = self._adjust_(iter)
            self.estimation_error_numer += math.fabs(error_rate) ** n
            self.estimation_error_denom += 1.0
            # error_rate = self.__adjust__(iter)
            # error_dist += math.pow(error_rate, 2.0)
        # end : for (decisions)

        error_rate = (
            0.0
            if self.estimation_error_denom == 0.0
            else (self.estimation_error_numer / self.estimation_error_denom) ** (1 / n)
        )

        return error_rate

    # end : private override float fit()

    ### member methods...
    def _preprocess_(self) -> None:
        # adjust_tags_score;
        for _ in range(self.EMIT_ITER_THRESHOLD):
            self._adjust_tags_score_()
        # end : for (iterations)

    # end : protected void preprocess()

    def _adjust_tags_score_(self) -> None:
        """태그점수를 보정합니다."""
        arr_weights = np.ones(
            shape=(
                self.tags_count,
                self.tags_count,
            )
        )
        for inst in self.view_list:
            inst: BaseAction
            denom = numer = estimated_score = error_rate = 0.0
            user: UserEntity = self.user_dict[inst.user_id]
            item: ItemEntity = self.item_dict[inst.item_id]

            # estimate
            for x_name in user.top_n_decision_tags_set:
                x_idx: int = self.tag_name_to_idx[x_name]
                for y_name in item.tags_set:
                    y_idx: int = self.tag_name_to_idx[y_name]
                    corr = self.arr_tags_score[x_idx][y_idx]
                    weight = arr_weights[x_idx][y_idx]
                    contains_tag = (
                        1
                        if y_name in user.top_n_decision_tags_set
                        else self._DEFAULT_VOTING
                    )
                    # contains_tag = 1 if y_name in user.top_n_decision_tags_set else 0
                    numer += corr * contains_tag * weight
                    denom += math.fabs(corr)
                # end : for (T(i))
            # end : for (NT(u))
            estimated_score = 0.0 if denom == 0.0 else numer / denom
            # is_fit ?
            if estimated_score == 1.0:
                continue

            # adjust score;
            error_rate = 1.0 - estimated_score
            for x_name in user.top_n_decision_tags_set:
                x_idx: int = self.tag_name_to_idx[x_name]
                for y_name in item.tags_set:
                    y_idx: int = self.tag_name_to_idx[y_name]
                    corr: float = self.arr_tags_score[x_idx][y_idx]
                    weight: float = arr_weights[x_idx][y_idx]
                    contains_tag: float = (
                        1
                        if y_name in user.top_n_decision_tags_set
                        else self._DEFAULT_VOTING
                    )
                    numer = 1.0 + self.LEARNING_RATE * weight * corr
                    denom = estimated_score + self.REGULARIZATION * corr
                    weight += error_rate * (numer / denom)
                # end : for (T(i))
            # end : for (NT(u))

    # end : protected void adjuste_tags_score()

    def _adjust_(self, inst: BaseAction) -> float:
        # if inst.estimated_score == 1.0:
        #     return 0.0
        error_rate = 1 - inst.estimated_score
        user: UserEntity = self.user_dict[inst.user_id]
        item: ItemEntity = self.item_dict[inst.item_id]
        uidx = self.user_id_to_idx[inst.user_id]
        update_score = self.LEARNING_RATE * error_rate
        for x_name in user.top_n_decision_tags_set:
            if not x_name in self.tag_name_to_idx:
                continue
            x_idx: int = self.tag_name_to_idx[x_name]
            for y_name in item.tags_set:
                if not y_name in self.tag_name_to_idx:
                    continue
                y_idx: int = self.tag_name_to_idx[y_name]
                corr: float = self.arr_tags_score[x_idx][y_idx]
                weight = self.arr_user_idx_to_weights[uidx][x_idx][y_idx]
                # weight = weight + update_score * weight
                ## [24.05.11] tags score ratio를 가하지 않았음 ;;;
                weight = math.tanh(weight + (update_score * weight * corr))
                self.arr_user_idx_to_weights[uidx][x_idx][y_idx] = weight
            # end : for (T(i))
        # end : for (NT(u))

        return error_rate

    # end : protected float adjust()

    # end :
    def _estimate_(self, inst: BaseAction) -> BaseAction:
        # inst = self.__adjusted_weighted_sum__(inst)
        inst = self.__weighted_sum__(inst)
        return inst

    # end : protected float estimate()

    def __weighted_sum__(
        self,
        inst: BaseAction,
    ) -> BaseAction:
        denom, numer = 0.0, 0.0
        user: UserEntity = self.user_dict[inst.user_id]
        item: ItemEntity = self.item_dict[inst.item_id]
        uidx = self.user_id_to_idx[user.user_id]
        for x_name in user.top_n_decision_tags_set:
            x_idx = self.tag_name_to_idx[x_name]
            for y_name in item.tags_set:
                y_idx = self.tag_name_to_idx[y_name]
                corr = self.arr_tags_score[x_idx][y_idx]
                weight = self.arr_user_idx_to_weights[uidx][x_idx][y_idx]
                # if np.isinf(weight) or np.isneginf(weight) or np.isnan(weight):
                #     print()
                contains_tag = (
                    1
                    if y_name in user.top_n_decision_tags_set
                    else self._DEFAULT_VOTING
                )
                # contains_tag = 1 if y_name in user.top_n_decision_tags_set else 0
                numer += corr * contains_tag * weight
                denom += math.fabs(corr)
            # end : for (T(i))
        # end : for (NT(u))
        inst.estimated_score = 0.0 if denom == 0.0 else numer / denom
        # if np.isnan(inst.estimated_score):
        #    print()
        return inst

    # end : private BaseAction weighted_sum()

    def __adjusted_weighted_sum__(
        self,
        inst: BaseAction,
    ) -> BaseAction:
        """
        - 요약: 사용안함
            - 추천하기 위한 전처리(학습까지) 1시간 걸림
            - 정확성, 효율 모두 떨어짐
        """
        users_reg, items_reg = 0.0, 0.0
        denom, numer = 0.0, 0.0
        user_u: UserEntity = self.user_dict[inst.user_id]
        item: ItemEntity = self.item_dict[inst.item_id]

        ### user: Equation (5)
        u_idx = self.user_id_to_idx[inst.user_id]
        denom = 0.0
        numer = 0.0
        for x_name in user_u.top_n_decision_tags_set:
            # if not x_name in self.tag_name_to_idx:
            #     continue
            x_idx: int = self.tag_name_to_idx[x_name]
            for y_name in item.tags_set:
                # if not y_name in self.tag_name_to_idx:
                #     continue
                y_idx: int = self.tag_name_to_idx[y_name]

                weight = self.arr_user_idx_to_weights[u_idx][x_idx][y_idx]
                co_occur = self.co_occur_ratio[x_idx][y_idx]
                score = self._item_based_pcc_tags_array[x_idx][y_idx]
                # score = self.arr_tags_score[x_idx][y_idx]

                # has_value = 1 if y_name in user_u.top_n_decision_tags_set else 0
                has_value = (
                    1
                    if y_name in user_u.top_n_decision_tags_set
                    else self._DEFAULT_VOTING
                )

                numer += weight * score * co_occur * has_value
                denom += math.fabs(weight * score)
            # end : for (T(i))
        # end : for (NT(u))
        users_reg = 0.0 if denom == 0.0 else numer / denom

        ### item: Equation (6)
        denom = 0.0
        numer = 0.0
        for user_id in item.set_of_like_user_ids:
            user_v: UserEntity = self.user_dict[user_id]
            v_idx: int = self.user_id_to_idx[user_id]
            for x_name in user_v.top_n_decision_tags_set:
                # if not x_name in self.tag_name_to_idx:
                #     continue
                x_idx: int = self.tag_name_to_idx[x_name]
                for y_name in item.tags_set:
                    # if not y_name in self.tag_name_to_idx:
                    #     continue
                    y_idx: int = self.tag_name_to_idx[y_name]

                    weight = self.arr_user_idx_to_weights[v_idx][x_idx][y_idx]
                    co_occur = self.co_occur_ratio[x_idx][y_idx]
                    # score = self.arr_tags_score[x_idx][y_idx]
                    score = self._user_based_pcc_tags_array[x_idx][y_idx]

                    has_value = (
                        1
                        if y_name in user_v.top_n_decision_tags_set
                        else self._DEFAULT_VOTING
                    )
                    # has_value = 1 if y_name in user_v.top_n_decision_tags_set else 0

                    numer += score * weight * co_occur * has_value
                    denom += math.fabs(weight * score) * co_occur
                # end : for (T(i))
            # end : for (NT(v))
            items_reg = 0.0 if denom == 0.0 else numer / denom
        # end : for (liked_users)
        inst.estimated_score = users_reg + items_reg
        return inst

    # end : private BaseAction adjusted_weighted_sum()


# end : class
