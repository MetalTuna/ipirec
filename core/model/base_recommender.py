from abc import *

from tqdm import tqdm

from .base_train import BaseTrain
from .base_dataset import BaseDataSet
from ..entity import BaseAction, UserEntity
from ..io import InstanceIO


# class BaseRecommender(metaclass=ABCMeta):
class BaseRecommender(InstanceIO):
    """
    - 요약:
        - 항목추천을 위한 추상클래스입니다.
    - 주의 (예측함수를 재정의할 경우):
        - 추천함수는 조건에 따라 예측함수가 호출된 후에 추천작업을 수행합니다.
            - 예측함수를 이전에 호출됐는지 여부확인 후, 미호출됐을 경우에만 작업을 수행하고, is_predicted = True한 후 종료해주세요.
    """

    def __init__(
        self,
        estimator: BaseTrain,
    ) -> None:
        self._is_predicted = False
        self._estimator = estimator

    # end : init()

    @abstractmethod
    def _preprocess_(self) -> None:
        """전처리"""
        raise NotImplementedError()

    def top_n_recommendation(
        self,
        top_n: int,
    ) -> None:
        """
        - 요약:
            - 상위 n개의 예측점수를 갖는 항목들로 추천목록을 작성합니다.

        - 매개변수:
            - top_n (int): 상위 n개에 대한 n값을 정합니다.
        """
        N = top_n - 1
        if not self._is_predicted:
            self.prediction()

        for user_id in self.user_dict.keys():
            user: UserEntity = self.user_dict[user_id]
            user.recommended_items_dict.clear()
            for idx in range(len(user.estimated_items_score_list)):
                if idx > N:
                    break
                inst: BaseAction = user.estimated_items_score_list[idx]
                user.recommended_items_dict.update({inst.item_id: inst.estimated_score})
            # end : for (candidate_items)
        # end : for (users)

    # end : public void top_n_recommendation()

    def threshold_recommendation(
        self,
        score_threshold: float,
    ) -> None:
        """
        - 요약:
            - 예측점수가 임계 값 이상인 항목들로 추천목록을 작성합니다.

        - 매개변수:
            - score_threshold (float): 예측점수의 임계 값을 정합니다.
        """
        if not self._is_predicted:
            self.prediction()
        for user_id in self.user_dict.keys():
            user: UserEntity = self.user_dict[user_id]
            user.recommended_items_dict.clear()
            for idx in range(len(user.estimated_items_score_list)):
                inst: BaseAction = user.estimated_items_score_list[idx]
                if inst.estimated_score < score_threshold:
                    break
                # end : if (threshold_condititon)
                user.recommended_items_dict.update({inst.item_id: inst.estimated_score})
            # end : for (candidate_items)
        # end : for (users)

    # end : public void threshold_recommendation()

    def prediction(self) -> None:
        """
        - 요약:
            - 각 사용자들의 항목들에 대한 의사결정 점수를 예측합니다. (사용자 단위)
        - 주의:
            - 이 함수를 재정의한다면, 반드시 이전의 예측작업 진행여부(self._is_predicted)를 확인하도록 구현하세요.
                - 실행됐다면, 진행여부에 반영하세요.

        >>> self._is_predicted = True
        """
        if self._is_predicted:
            return
        self._preprocess_()
        # for user_id in self.user_dict.keys():
        for user_id in tqdm(
            iterable=self.user_dict.keys(),
            desc="Recommender.prediction()",
            total=self.users_count,
        ):
            user: UserEntity = self.user_dict[user_id]
            predicts_list = list()
            for item_id in user.candidate_item_ids_set:
                inst = BaseAction(user_id, item_id)
                inst.estimated_score = self.predict(user_id, item_id)
                predicts_list.append(inst)
            # end : for (items)

            # 항목의 예측결과는 list로 채우고 내림차순으로 정렬해서 만든다.
            user.estimated_items_score_list = sorted(
                predicts_list, key=lambda x: x.estimated_score, reverse=True
            )
            # 추천목록은 선별된 항목들로 set 또는 dict로 만든다. 이 기능은 다른 함수에서~~
        # end : for (users)
        self._is_predicted = True

    # end : public void recommendation()

    def predict(
        self,
        user_id: int,
        item_id: int,
    ) -> float:
        """
        요약:
            사용자의 항목에 대한 의사결정 점수를 예측합니다. (원소단위)

        Args:
            user_id (int): _description_
            item_id (int): _description_

        Returns:
            float: _description_
        """
        return self._estimator.predict(
            user_id,
            item_id,
        )

    # end : public void prediction()

    ### properties
    @property
    def _dataset(self) -> BaseDataSet:
        return self._estimator._dataset

    ## direct references properties
    @property
    def user_dict(self) -> dict:
        return self._estimator._dataset.user_dict

    @property
    def item_dict(self) -> dict:
        return self._estimator._dataset.item_dict

    @property
    def tags_dict(self) -> dict:
        return self._estimator._dataset.tags_dict

    @property
    def users_count(self) -> int:
        return self._estimator._dataset.users_count

    @property
    def items_count(self) -> int:
        return self._estimator._dataset.items_count

    @property
    def tags_count(self) -> int:
        return self._estimator._dataset.tags_count

    @property
    def view_list(self) -> list:
        return self._estimator._dataset.view_list

    @property
    def like_list(self) -> list:
        return self._estimator._dataset.like_list

    @property
    def purchase_list(self) -> list:
        return self._estimator._dataset.purchase_list

    # index reference properties
    @property
    def user_id_to_idx(self) -> dict:
        return self._estimator.user_id_to_idx

    @property
    def user_idx_to_id(self) -> dict:
        return self._estimator.user_idx_to_id

    @property
    def item_id_to_idx(self) -> dict:
        return self._estimator.item_id_to_idx

    @property
    def item_idx_to_id(self) -> dict:
        return self._estimator.item_idx_to_id

    @property
    def tag_name_to_idx(self) -> dict:
        return self._estimator._dataset.tag_name_to_idx

    @property
    def tag_idx_to_name(self) -> dict:
        return self._estimator._dataset.tag_idx_to_name

    @property
    def _recommender_name(self) -> str:
        return type(self).__name__

    @property
    def _config_info(self):
        return self._estimator._config_info


# end : class
