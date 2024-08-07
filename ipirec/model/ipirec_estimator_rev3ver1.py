"""
[작성일] 24.07.09 17:29. 마지막 분석 (구현 중)
[수정일] 
- 24.07.10 10:00. 
"""

import os
import sys
import copy
import gc
import pickle

import numpy as np

# [DEF] Environment variables
__dir_name__ = os.path.basename(os.path.dirname(__file__))
WORKSPACE_HOME = os.path.dirname(__file__).replace(f"/ipirec/{__dir_name__}", "")
""".../ipirec"""
DATA_SET_HOME = f"{WORKSPACE_HOME}/data/colley"
""">>> `${WORKSPACE_HOME}`/data"""

sys.path.append(WORKSPACE_HOME)

from core import *
from colley import *
from .ipirec_estimator_rev3 import IPIRecEstimatorRev3


class IPIRecEstimatorRev3Ver1(IPIRecEstimatorRev3):
    """
    - 요약:
        - 일반화 항이 변경됨
    """

    def __init__(
        self,
        model: BaseModel,
        model_params: dict,
    ) -> None:
        super().__init__(
            model,
            model_params,
        )

    # end : init()

    def _generalization_term_(self) -> float:
        __F = __W = __S = _U = 0.0
        """
        for user_id in self.user_dict.keys():
            uidx: int = self.user_id_to_idx[user_id]
            _x_idxs: set = self._user_id_to_tags_idx[user_id]
            _x_len = len(_x_idxs)
            if _x_len == 0:
                continue
            _U += 1
            _w = 0.0
            _f = 0.0
            for _x in _x_idxs:
                for _y in self.tag_idx_to_name.keys():
                    __w = self.arr_user_idx_to_weights[uidx][_x][_y]
                    __s = self.arr_tags_score[_x][_y]
                    __f = (__w * __s) ** 2.0
                    __w = __w**2.0
                    _w += __w
                    _f += __f**2.0
            _w = (_w / (2 * self.tags_count * _x_len)) ** 0.5
            __W += _w
        __W = __W / _U
        """
        for user_id in self.user_dict.keys():
            uidx: int = self.user_id_to_idx.get(user_id)
            _x_idxs: set = self._user_id_to_tags_idx.get(user_id)
            _x_len = len(_x_idxs)
            if _x_len == 0:
                continue
            _ws = 0.0
            _uw = 0.0
            for _x in _x_idxs:
                _w = 0.0
                for _y in self.tag_idx_to_name.keys():
                    __w = self.arr_user_idx_to_weights[uidx][_x][_y]
                    __s = self.arr_tags_score[_x][_y]
                    _w += __w**2.0
                    _ws += (__w * __s) ** 2.0
                _uw += _w / self.tags_count
            __W += _uw / (2 * _x_len)
            __F += (_ws / (_x_len * self.tags_count)) ** (1 / 2)
            _U += 1.0
        for _x in self.tag_idx_to_name.keys():
            for _y in self.tag_idx_to_name.keys():
                __S += self.arr_tags_score[_x][_y] ** 2
        __S = (1 / (2 * self.tags_count)) * (__S**0.5)
        _G = (len(self.view_list) + len(self.like_list) + len(self.purchase_list)) / (
            _U * self.items_count
        )
        __W = __W / _U
        __F = 0.0 if _U == 0.0 else (__F / _U)
        __REG = _G * (__F + __W + __S)
        return __REG

    # end : protected override float generalization_term()

    """
    def _generalization_term_(self) -> float:
        # [24.07.11] 3090에서 잘못 실행됨 (코드 동기화가 안됐음)
        _numer = _denom = 0.0
        _T_IDXs = list(self.tag_idx_to_name.keys())
        for user_id in self.user_dict.keys():
            uidx: int = self.user_id_to_idx.get(user_id)
            _x_idxs: set = self._user_id_to_tags_idx.get(user_id)
            _x_len = len(_x_idxs)

            if _x_len == 0:
                continue
            __ws = 0.0
            for x_idx in _x_idxs:
                for y_idx in _T_IDXs:
                    __ws += (
                        self.arr_tags_score[x_idx][y_idx]
                        - (
                            self.arr_user_idx_to_weights[uidx][x_idx][y_idx]
                            * self.arr_tags_score[x_idx][y_idx]
                        )
                    ) ** 2.0
                # end : for (forall T)
            # end : for (T(u))
            _numer += (__ws / (_x_len * self.tags_count)) ** (1 / 2)
            _denom += 1.0
            # _W: np.ndarray = self.arr_user_idx_to_weights[uidx]
            # self.arr_tags_score
        # end : for (users)

        _G = (len(self.view_list) + len(self.like_list) + len(self.purchase_list)) / (
            _denom * self.items_count
        )
        __REG = 0.0 if _denom == 0.0 else _G * (_numer / _denom)

        return __REG

    # end : protected override float generalization_term()
    """


# end : class
