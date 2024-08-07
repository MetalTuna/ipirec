"""
[작성일] 24.05.14.
[수정일] 
- 24.06.04. 태그점수 보정할 때, 의사결정 종류를 선택할 수 있도록 변경됨
- 24.05.16.
    - 태그점수를 보정할 때, 분모가 0에 수렴해서 보정정도 값이 극한을 갖게됨
    - 그래서 분모를 가중치 값이 아닌 사용된 빈도 수를 구하도록 변경
        - 참여태그의 점수*가중치 합의 평균이 다음 가중치 보정으로 참여

TagsCorrelationEstimator 
"""

import numpy as np
from tqdm import tqdm

from core import *
from ..models.ipirec_model_series2 import IPIRecModelSeries2


class BaseCorrelationEstimator(BaseTrain):
    """
    - 요약:
        - 태그점수 보정을 다루는 클래스입니다. -- AdjustTagsCorrelation()
            - 모든 봤다 내역을 잘 맞출 수 있는 태그점수 S를 만들기 위한 목적
            - S를 보정하는 가중치 W를 구한 후, S = S * W로 태그점수를 보정합니다.
    """

    def __init__(
        self,
        model: BaseModel,
        model_params: dict,
    ) -> None:
        self.ACTUAL_SCORE = 1.0
        super().__init__(model, model_params)

    # end : init()

    ## member variables
    # properties
    @property
    def model(self) -> IPIRecModelSeries2:
        return self._model

    @property
    def arr_tags_score(self) -> np.ndarray:
        return self.model.arr_tags_score

    ## member functions...
    def _contains_score_(
        self,
        user: UserEntity,
        tag_name: str,
    ) -> float:
        """
        - 요약:
            - 사용자 의사결정 상위 N개의 태그집합에 미포함된 항목의 태그점수 산정을 위한 함수입니다.

        - 매개변수:
            - user (UserEntity): 목표 사용자 객체 인스턴스입니다.
            - tag_name (str): 항목의 태그 원소입니다.

        - 반환:
            float: 사용자의 상위 N개의 태그에 포함되면 1.0, 아니면 기본 값(default_voting)을 반환합니다.
        """
        return 1.0 if tag_name in user.top_n_decision_tags_set else self.default_voting

    # end : protected float contains_score()

    def _preprocess_(self):
        self.adjust_tags_score()

    # end : protected override Any preprocess()

    def _adjust_(self, inst: BaseAction) -> float:
        """
        - 요약:
            - 태그점수 보정 (개인화)
        """
        if inst.estimated_score == self.ACTUAL_SCORE:
            return 0.0

        user: UserEntity = self.user_dict[inst.user_id]
        item: ItemEntity = self.item_dict[inst.item_id]
        error_rate = self.ACTUAL_SCORE - inst.estimated_score
        uidx = self.user_id_to_idx[inst.user_id]
        numer = denom = 0.0

        for x_name in user.top_n_decision_tags_set:
            x_idx: int = self.tag_name_to_idx[x_name]
            for y_name in item.tags_set:
                y_idx: int = self.tag_name_to_idx[y_name]
                if x_idx == y_idx:
                    continue
                denom = 0.0
                for tag_name in item.tags_set:
                    idx: int = self.tag_name_to_idx[tag_name]
                    if idx == x_idx:
                        continue
                    denom += np.fabs(self.arr_user_idx_to_weights[uidx][x_idx][idx])
                # end : for (T(i))
                denom = inst.estimated_score + (self.REGULARIZATION * denom)

                weight = self.arr_user_idx_to_weights[uidx][x_idx][y_idx]
                corr = self.arr_tags_score[x_idx][y_idx]
                numer = self.ACTUAL_SCORE + (self.LEARNING_RATE * corr)
                weight = weight * (numer / denom)
                self.arr_user_idx_to_weights[uidx][x_idx][y_idx] = weight
            # end : for (T(i))
        # end : for (NT(u))
        return error_rate**self.frob_norm

    # end : protected override float adjust()

    def adjust_tags_score(self) -> None:
        desc_str = str.format(
            "{0}.adjust_tags_score()",
            type(self).__name__,
        )
        for _ in tqdm(
            iterable=range(self.score_iterations),
            desc=desc_str,
            total=self.score_iterations,
        ):
            self.estimation_error_denom = 0.0
            self.estimation_error_numer = 0.0
            self._adjust_tags_corr_()
            self._current_mean_of_error_scores = (
                0.0
                if self.estimation_error_denom == 0.0
                else (self.estimation_error_numer / self.estimation_error_denom)
                ** (1 / self.frob_norm)
            )
        # end : for (score_iters)

    # end : public void adjust_tags_score()

    def _adjust_tags_corr_(
        self,
        decision_type: DecisionType = DecisionType.E_VIEW,
    ):
        """
        - 요약:
            - 태그점수 보정 (전역)
        """
        arr_weights = np.ones(
            shape=(
                self.tags_count,
                self.tags_count,
            ),
        )
        _kwd: str = DecisionType.to_str(decision_type)
        _decisions_list: list = self._dataset._decision_dict[_kwd]

        self.estimation_error_denom = 0.0
        self.estimation_error_numer = 0.0

        for inst in _decisions_list:
            inst: BaseAction
            user: UserEntity = self.user_dict[inst.user_id]
            item: ItemEntity = self.item_dict[inst.item_id]

            ## estimation
            numer = denom = weighted_score = 0.0
            for x_name in user.top_n_decision_tags_set:
                x_idx: int = self.tag_name_to_idx[x_name]
                for y_name in item.tags_set:
                    y_idx: int = self.tag_name_to_idx[y_name]
                    if x_idx == y_idx:
                        continue
                    corr = self.arr_tags_score[x_idx][y_idx]
                    weight = arr_weights[x_idx][y_idx]
                    contains_score = self._contains_score_(
                        user=user,
                        tag_name=y_name,
                    )
                    weighted_score = corr * weight
                    numer += weighted_score * contains_score
                    denom += np.fabs(weighted_score)
                # end : (T(i))
            # end : for (NT(u))
            estimated_score = 0.0 if denom == 0.0 else numer / denom

            # is_estimated?
            if denom == 0.0:
                continue

            self.estimation_error_denom += 1.0
            # is_fit?
            if estimated_score == self.ACTUAL_SCORE:
                continue

            ## feedback
            error_rate = self.ACTUAL_SCORE - estimated_score
            self.estimation_error_numer += error_rate**self.frob_norm
            for x_name in user.top_n_decision_tags_set:
                for y_name in item.tags_set:
                    corr = self.arr_tags_score[x_idx][y_idx]
                    weight = arr_weights[x_idx][y_idx]
                    eta = (
                        error_rate
                        * self.score_learning_rate
                        * self.score_generalization
                        * corr
                    )
                    eta = 0.5 * eta if eta > 0 else 2 * eta
                    ## 가중치가 변경된만큼 분모 값도 변하므로,
                    # 가중치 변경정도를 총합에 가한 후의 값을 분모로 사용
                    denom = 0.0
                    for tag_name in item.tags_set:
                        idx: int = self.tag_name_to_idx[tag_name]
                        if idx == x_idx:
                            continue
                        denom += np.fabs(
                            self.arr_tags_score[x_idx][idx] * arr_weights[x_idx][idx]
                        )
                    # end : for (T(i))

                    if np.isnan(denom):
                        denom = 0.0
                    if denom == 0.0:
                        continue

                    numer = (weight + eta) / denom
                    denom += weight - numer
                    arr_weights[x_idx][y_idx] = numer
                # end : for (T(i))
            # end : for (NT(u))

            self._current_mean_of_error_scores = (
                0.0
                if self.estimation_error_denom == 0.0
                else (self.estimation_error_numer / self.estimation_error_denom)
                ** (1 / self.frob_norm)
            )
        # end : for (decisions_list)

        self.model.arr_tags_score = np.multiply(
            self.arr_tags_score,
            arr_weights,
        )

    # end : protected Any adjust_tags_corr()

    def _estimate_(self, inst: BaseAction) -> BaseAction:
        """weighted_sum"""
        user: UserEntity = self.user_dict[inst.user_id]
        item: ItemEntity = self.item_dict[inst.item_id]
        uidx = self.user_id_to_idx[user.user_id]
        numer = denom = 0.0
        for x_name in user.top_n_decision_tags_set:
            x_idx: int = self.tag_name_to_idx[x_name]
            for y_name in item.tags_set:
                y_idx: int = self.tag_name_to_idx[y_name]
                if x_idx == y_idx:
                    continue
                corr = self.arr_tags_score[x_idx][y_idx]
                weight = self.arr_user_idx_to_weights[uidx][x_idx][y_idx]
                contains_score = self._contains_score_(
                    user=user,
                    tag_name=y_name,
                )
                weighted_score = corr * weight
                numer += weighted_score * contains_score
                denom += np.fabs(weighted_score)
            # end : for (T(i))
        # end : for (NT(u))
        inst.estimated_score = 0.0 if denom == 0.0 else numer / denom
        return inst

    # end : protected override BaseAction estimate()

    @staticmethod
    def create_models_parameters(
        score_iterations: int,
        score_learning_rate: float,
        score_generalization: float,
        weight_iterations: int,
        weight_learning_rate: float,
        weight_generalization: float,
        frob_norm: int = 1,
        default_voting: float = 0.0,
    ) -> dict:
        return {
            "score_iterations": score_iterations,
            "score_learning_rate": score_learning_rate,
            "score_generalization": score_generalization,
            "weight_iterations": weight_iterations,
            "weight_learning_rate": weight_learning_rate,
            "weight_generalization": weight_generalization,
            "frob_norm": frob_norm,
            "default_voting": default_voting,
        }

    # end : public static override dict create_models_parameters()

    def _set_model_params_(self, model_params: dict) -> None:
        self.score_iterations = model_params["score_iterations"]
        self.score_learning_rate = model_params["score_learning_rate"]
        self.score_generalization = model_params["score_generalization"]
        self.EMIT_ITER_THRESHOLD = model_params["weight_iterations"]
        self.LEARNING_RATE = model_params["weight_learning_rate"]
        self.REGULARIZATION = model_params["weight_generalization"]
        self.frob_norm = model_params["frob_norm"]
        self.default_voting = model_params["default_voting"]

    # end : protected override void set_model_params()


# end : class
