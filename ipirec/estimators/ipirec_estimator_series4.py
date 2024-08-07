"""
[작성일] 24.07.09 17:29. 마지막 분석 (구현 중)
[수정일] 
- 24.07.10 10:00. 
"""

# built-in
import os
import sys

# [DEF] Environment variables
__dir_name__ = os.path.basename(os.path.dirname(__file__))
WORKSPACE_HOME = os.path.dirname(__file__).replace(f"/ipirec/{__dir_name__}", "")
""".../ipirec"""
DATA_SET_HOME = f"{WORKSPACE_HOME}/data/colley"
""">>> `${WORKSPACE_HOME}`/data"""
sys.path.append(WORKSPACE_HOME)

# custom LIB.
from core import *
from colley import *
from .ipirec_estimator_series3 import IPIRecEstimatorSeries3


class IPIRecEstimatorSeries4(IPIRecEstimatorSeries3):
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

        for user_id in self.user_dict.keys():
            uidx: int = self.user_id_to_idx.get(user_id)
            _x_idxs: set = self._user_id_to_tags_idx.get(user_id)
            _x_len = len(_x_idxs)
            _ws = 0.0
            _uw = 0.0

            if _x_len == 0:
                continue

            for _x in _x_idxs:
                _w = 0.0
                for _y in self.tag_idx_to_name.keys():
                    __w = self.arr_user_idx_to_weights[uidx][_x][_y]
                    __s = self.arr_tags_score[_x][_y]
                    _w += __w**2.0
                    _ws += (__w * __s) ** 2.0
                # end : for (T(i))
                _uw += _w / self.tags_count
            # end : for (T(u))

            __W += _uw / (2 * _x_len)
            __F += (_ws / (_x_len * self.tags_count)) ** (1 / 2)
            _U += 1.0
        # end : for (users)

        for _x in self.tag_idx_to_name.keys():
            for _y in self.tag_idx_to_name.keys():
                __S += self.arr_tags_score[_x][_y] ** 2
            # end : for (T(i))
        # end : for (T(u))

        __S = (1 / (2 * self.tags_count)) * (__S**0.5)
        _G = (len(self.view_list) + len(self.like_list) + len(self.purchase_list)) / (
            _U * self.items_count
        )
        __W = __W / _U
        __F = 0.0 if _U == 0.0 else (__F / _U)
        __REG = _G * (__F + __W + __S)

        return __REG

    # end : protected override float generalization_term()


# end : class
