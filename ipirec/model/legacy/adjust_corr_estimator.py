"""
[작성일] 24.05.20.
- 태그점수 보정에 변화를 줌
    - 전역: 태그점수를 직접 변경
    - 개인: 보정 값이 가해진 재예측 결과에 대한 오차정도를 다음 변수에 가하도록 수정 
[수정사항]
- 24.05.22 10:55. 예측이 너무 오래걸려서, 예측함수에서 numpy함수를 사용하지 않도록 변경함
- 24.05.21 16:53. reshape로 인한 bottleneck확인 및 제거를 위해 해당부분들을 수정함
- 24.05.21 15:58. 구상안대로 구현했음 
"""

import numpy as np

from core import BaseModel, BaseAction, UserEntity, ItemEntity, DecisionType
from .base_corr_estimator import BaseCorrelationEstimator


class AdjustCorrelationEstimator(BaseCorrelationEstimator):

    def __init__(
        self,
        model: BaseModel,
        model_params: dict,
    ) -> None:
        super().__init__(
            model,
            model_params,
        )
        self.arr_users_tags_map = np.zeros(
            shape=(
                self.users_count,
                self.tags_count,
            ),
            dtype=np.int8,
        )
        """
        - Binary array (Users X tags)
        """
        self.arr_items_tags_map = np.zeros(
            shape=(
                self.items_count,
                self.tags_count,
            ),
            dtype=np.int8,
        )
        """
        - Binary Array (Items X tags)
        """

        self.user_id_to_tags_map_dict = dict()
        """
        - 요약:
            - 사용자들의 상위 N개의 선호태그에 대한 이진행렬 사전입니다.
        - 구성:
            - Key: user_id (int)
            - Value: binary_array (np.int8): |T| * 1
        """
        self.item_id_to_tags_map_dict = dict()
        """
        - 요약:
            - 항목들이 속하는 태그에 대한 이진행렬 사전입니다.
            - 항목은 전치행렬만 다루므로, 전치된 행렬로 구성합니다.
        - 구성:
            - Key: item_id (int)
            - Value: binary_array (np.int8): 1 * |T|
        """

        self.__build_binary_tags_array__()

    # end : init()

    def __build_binary_tags_array__(self) -> None:
        """이진행렬 만들기"""
        for user_id in self.user_dict.keys():
            user: UserEntity = self.user_dict[user_id]
            uidx: int = self.user_id_to_idx[user_id]
            for tag_name in user.top_n_decision_tags_set:
                tidx: int = self.tag_name_to_idx[tag_name]
                self.arr_users_tags_map[uidx][tidx] = 1
            # end : for (NT(u))
            self.user_id_to_tags_map_dict.update(
                {user_id: self.arr_users_tags_map[uidx].reshape(self.tags_count, 1)}
            )
        # end : for (users)

        for item_id in self.item_dict.keys():
            if not item_id in self.item_id_to_idx:
                continue
            item: ItemEntity = self.item_dict[item_id]
            iidx: int = self.item_id_to_idx[item_id]
            for tag_name in item.tags_set:
                tidx: int = self.tag_name_to_idx[tag_name]
                self.arr_items_tags_map[iidx][tidx] = 1
            # end : for (T(i))
            self.item_id_to_tags_map_dict.update(
                {item_id: self.arr_items_tags_map[iidx].reshape(self.tags_count, 1).T}
            )
        # end : for (items)

    # end : private void build_binary_tags_array()

    def _estimate_(self, inst: BaseAction) -> BaseAction:
        """
        - GPU 사용못하는 것 같으니 일단 잠근다;
        uidx: int = self.user_id_to_idx[inst.user_id]
        _U: np.ndarray = self.user_id_to_tags_map_dict[inst.user_id]
        _I: np.ndarray = self.item_id_to_tags_map_dict[inst.item_id]

        - 행렬곱
        _F = np.matmul(_U, _I)
            - feature_map << top_n_tags_map (|T| * |T|)
        _C = np.matmul(1 * np.logical_not(_U == 1), _I)
            - contains_map << item_tags_map (|T| * |T|)

        >>> _FW = self.arr_user_idx_to_weights[uidx] * _F
        >>> _FS = _FW * self.arr_tags_score
        >>> _DS = self.default_voting * _C

        >>> numer = np.sum(_FS + _DS)
        >>> denom = np.sum(np.fabs(_FS))
        """

        ## ㅇㅇ 이대로 ㄱㄱ
        # 모든 항목의 점수예측에 numpy는 30분, 이렇게 하면 1분 걸림
        user: UserEntity = self.user_dict[inst.user_id]
        item: ItemEntity = self.item_dict[inst.item_id]
        uidx: int = self.user_id_to_idx[inst.user_id]
        numer = denom = 0.0

        for x_name in user.top_n_decision_tags_set:
            x_idx: int = self.tag_name_to_idx[x_name]
            for y_name in item.tags_set:
                y_idx: int = self.tag_name_to_idx[y_name]
                if x_idx == y_idx:
                    continue
                _weight = self.arr_user_idx_to_weights[uidx][x_idx][y_idx]
                _corr = self.arr_tags_score[x_idx][y_idx]
                _score = _weight * _corr

                numer += self._contains_score_(user, y_name) * _score
                denom += np.fabs(_weight)
            # end : for (T(i))
        # end : for (NT(u))

        inst.estimated_score = 0.0 if denom == 0.0 else numer / denom
        return inst

    # end : protected override BaseAction estimate()

    def _adjust_tags_corr_(
        self,
        decision_type: DecisionType = DecisionType.E_VIEW,
    ) -> float:
        """adjust tags score"""
        _kwd = DecisionType.to_str(decision_type)
        _decision_list: list = self._dataset._decision_dict[_kwd]
        _err_dist = 0.0

        for inst in _decision_list:
            inst: BaseAction
            user: UserEntity = self.user_dict[inst.user_id]
            item: ItemEntity = self.item_dict[inst.item_id]
            _U: np.ndarray = self.user_id_to_tags_map_dict[inst.user_id]
            _I: np.ndarray = self.item_id_to_tags_map_dict[inst.item_id]

            ## 행렬곱
            _F = np.matmul(_U, _I)
            """feature_map << top_n_tags_map (|T| * |T|)"""
            # _C = np.matmul(1 * np.logical_not(_U == 1), _I)
            # """contains_map << item_tags_map (|T| * |T|)"""

            _is_fit = False
            for x_name in user.top_n_decision_tags_set:
                if _is_fit:
                    break
                x_idx: int = self.tag_name_to_idx[x_name]
                for y_name in item.tags_set:
                    y_idx: int = self.tag_name_to_idx[y_name]
                    if x_idx == y_idx:
                        continue
                    """
                    # estimate
                    _FS = self.arr_tags_score * _F
                    _DS = self.default_voting * _C
                    _E = _FS + _DS
                    numer = np.sum(_E)
                    denom = np.sum(np.fabs(_F))
                    ## 24.05.28. 19:22 -- typo;;
                    # denom = np.sum(np.fabs(_FS))
                    # inst.estimated_score = 0.0 if denom == 0.0 else numer / denom
                    """
                    inst = self._estimate_(inst)

                    # is_fit?
                    if inst.estimated_score == 1.0:
                        _is_fit = True
                        break

                    ## feedback
                    _err = self.ACTUAL_SCORE - inst.estimated_score
                    _corr = self.arr_tags_score[x_idx][y_idx]
                    _FS = self.arr_tags_score * _F
                    denom = np.sum(np.fabs(_FS))
                    _x = 0.0 if denom == 0.0 else (_corr / denom) * _err
                    _x = _x / 2 if _x >= 0 else _x * 2
                    _x = self.score_learning_rate * _x
                    _b = self.score_generalization * (
                        (np.sum(_FS[x_idx][:]) + np.sum(_FS[:][y_idx])) / 2
                    )
                    self.arr_tags_score[x_idx][y_idx] = _corr + _x + _b

                    # numer = (self.ACTUAL_SCORE + __adjust_score) +
                # end : for (T(i))
            # end : for (NT(u))
            inst = self._estimate_(inst)
            _err_dist += np.fabs(self.ACTUAL_SCORE - inst.estimated_score)
        # end : for (decisions_list)

        return _err_dist

    # end : protected override float adjust_tags_corr()

    def _adjust_(self, inst: BaseAction) -> float:

        uidx: int = self.user_id_to_idx[inst.user_id]
        user: UserEntity = self.user_dict[inst.user_id]
        item: ItemEntity = self.item_dict[inst.item_id]
        _U: np.ndarray = self.user_id_to_tags_map_dict[inst.user_id]
        _I: np.ndarray = self.item_id_to_tags_map_dict[inst.item_id]

        ## 행렬곱
        _F = np.matmul(_U, _I)
        """feature_map << top_n_tags_map (|T| * |T|)"""
        # _C = np.matmul(1 * np.logical_not(_U == 1), _I)
        # """contains_map << item_tags_map (|T| * |T|)"""

        ### update feedback
        # _is_fit = False
        for x_name in user.top_n_decision_tags_set:
            x_idx: int = self.tag_name_to_idx[x_name]
            # if _is_fit:
            #     break
            for y_name in item.tags_set:
                y_idx: int = self.tag_name_to_idx[y_name]
                if x_idx == y_idx:
                    continue

                ### Feedback
                ## User distance
                # estimate
                """
                _FW = self.arr_user_idx_to_weights[uidx] * _F
                _FS = self.arr_tags_score * _F
                _WS = _FW * _FS
                # _FS = _FW * self.arr_tags_score
                _DS = self.default_voting * _C

                numer = np.sum(_WS + _DS)
                denom = np.sum(np.fabs(_WS))
                inst.estimated_score = 0.0 if denom == 0.0 else numer / denom
                """
                inst = self._estimate_(inst)

                # if_fit ?
                if inst.estimated_score == 1.0:
                    # _is_fit = True
                    break

                # Adjust distance
                _FW = self.arr_user_idx_to_weights[uidx] * _F
                _adjust = 1.0 if inst.estimated_score == 0.0 else 0.0
                _weight = self.arr_user_idx_to_weights[uidx][x_idx][y_idx]
                _corr = self.arr_tags_score[x_idx][y_idx]
                numer = (self.ACTUAL_SCORE + _adjust) + (self.LEARNING_RATE * _corr)
                denom = (inst.estimated_score + _adjust) + (
                    self.REGULARIZATION * np.sum(np.fabs(_FW))
                )
                _weight = _weight * (numer / denom)
                self.arr_user_idx_to_weights[uidx][x_idx][y_idx] = _weight

                ## Tag score
                # estimate
                """
                _FW = self.arr_user_idx_to_weights[uidx] * _F
                _FS = self.arr_tags_score * _F
                _WS = _FW * _FS
                # _FS = _FW * self.arr_tags_score
                _DS = self.default_voting * _C

                numer = np.sum(_WS + _DS)
                denom = np.sum(np.fabs(_WS))
                inst.estimated_score = 0.0 if denom == 0.0 else numer / denom
                """
                inst = self._estimate_(inst)

                # if_fit ?
                if inst.estimated_score == 1.0:
                    # _is_fit = True
                    break

                # Adjust score
                _FS = self.arr_tags_score * _F
                _adjust = 1.0 if inst.estimated_score == 0.0 else 0.0
                _weight = self.arr_user_idx_to_weights[uidx][x_idx][y_idx]
                _corr = self.arr_tags_score[x_idx][y_idx]
                numer = (self.ACTUAL_SCORE + _adjust) + (self.LEARNING_RATE * _weight)
                denom = (inst.estimated_score + _adjust) + (
                    self.REGULARIZATION * np.sum(np.fabs(_FS))
                )
                ## [24.05.23] typo - logic err
                _score = _corr * (numer / denom)
                self.arr_tags_score[x_idx][y_idx] = _score
            # end : for (T(i))
        # end : for (NT(u))

        inst = self._estimate_(inst)
        return (self.ACTUAL_SCORE - inst.estimated_score) ** self.frob_norm

    # end : protected override float adjust()


# end : class
