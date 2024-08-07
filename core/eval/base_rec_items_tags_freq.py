from abc import *
import os

from ..entity import UserEntity, ItemEntity, BaseAction
from ..model import BaseRecommender


class BaseRecommenderItemsTagsFreq(metaclass=ABCMeta):
    """
    - 요약:
        - "추천된 항목들의 태그 출현 빈도 수"가 "검증 데이터에 항목들의 태그 출현 빈도 수"와 일치했는지 비교합니다.
        - `tags_freq_distance()`를 호출해서 결과를 구합니다.
    """

    def __init__(
        self,
    ) -> None:
        # member variables
        self._rec_tags_freq_dict = dict()
        """
        - 요약:
            - 추천된 항목들이 속한 태그들의 발생빈도 수를 집계합니다.
        - 구성:
            - Key: tag_name (str)
            - Value: freq (int)
        """
        self._test_tags_freq_dict = dict()
        """
        - 요약:
            - 검증 데이터의 항목들이 속한 태그들의 발생빈도 수를 집계합니다.
        - 구성:
            - Key: tag_name (str)
            - Value: freq (int)
        """

        # references instances
        self._recommender: BaseRecommender = None
        """
        - 요약:
            - 추천기를 참조해 추천 결과를 사용합니다.
        """
        self._test_set_list: list = None
        """
        - 요약:
            - 검증 데이터에 속한 의사결정 목록입니다. - BaseAction
            - 추천기와 쌍으로 참조합니다.
        """

    # end : init()

    def tags_freq_distance(
        self,
        test_set: list,
        recommender: BaseRecommender,
    ) -> float:
        """
        - 요약:
            - 검증데이터와 추천결과에 대한 항목들의 태그 출현빈도를 구한 후, 두 태그 분포의 거리를 구합니다.

        - 매개변수:
            test_set (list): 검증 데이터 셋
            recommender (BaseRecommender): 추천이 실행된 추천모델 객체

        - 반환:
            float: 두 태그 분포의 거리를 0~1의 값으로 출력합니다.
        """
        if isinstance(test_set, list):
            self._test_set_list = test_set
        elif isinstance(test_set, str):
            if not os.path.exists(test_set):
                raise FileNotFoundError()
            self._test_set_list: list = BaseAction.load_collection(
                file_path=test_set,
            )
        else:
            raise ValueError()

        self._recommender = recommender
        self._aggregation_()
        return self._distance_()

    # end : public float tags_freq_distance()

    @abstractmethod
    def _distance_(self) -> float:
        """일단 이건 코사인으로만 볼 예정 (다른거 보는게 무의미하기도 함)"""
        raise NotImplementedError()

    # protected float distance()

    def _aggregation_(self) -> None:
        rec_freq = dict()
        test_freq = dict()

        # 태그 빈도 수 집계 (검증집합)
        for inst in self._test_set_list:
            inst: BaseAction
            item: ItemEntity = self._item_dict[inst.item_id]
            for tag_name in item.tags_set:
                if not tag_name in test_freq:
                    test_freq.update({tag_name: 0})
                test_freq[tag_name] += 1
            # end : for (T(i))
        # end : for (test-set)

        # 태그 빈도 수 집계 (추천결과)
        for user_id in self._user_dict.keys():
            user: UserEntity = self._user_dict[user_id]
            for item_id in user.recommended_items_dict.keys():
                item: ItemEntity = self._item_dict[item_id]
                for tag_name in item.tags_set:
                    if not tag_name in rec_freq:
                        rec_freq.update({tag_name: 0})
                    rec_freq[tag_name] += 1
                # end : for (T(i))
            # end : for (rec_items)
        # end : for (users)

        self._rec_tags_freq_dict = rec_freq
        self._test_tags_freq_dict = test_freq

    # end : protected void aggregation()

    ##  properties
    # protected:
    @property
    def _user_dict(self) -> dict:
        """
        - 요약:
            - 추천기를 참조해 사용자 사전을 사용합니다.
        - 구성:
            - Key: user_id (int)
            - Value: inst (UserEntity)
        """
        return self._recommender._estimator._model._dataset.user_dict

    @property
    def _item_dict(self) -> dict:
        """
        - 요약:
            - 추천기를 참조해 항목 사전을 사용합니다.
        - 구성:
            - Key: item_id (int)
            - Value: inst (ItemEntity)
        """
        return self._recommender._estimator._model._dataset.item_dict

    @property
    def _tags_dict(self) -> dict:
        """
        - 요약:
            - 추천기를 참조해 태그 사전을 사용합니다.
        - 구성:
            - Key: tag_name (str)
            - Value: inst (TagEntity)
        """
        return self._recommender._estimator._model._dataset.tags_dict

    @property
    def _tag_name_to_idx(self) -> dict:
        """
        - 요약:
            - 추천기를 참조해 태그 사전을 사용합니다.
        - 구성:
            - Key: tag_name (str)
            - Value: index (int)
        """
        return self._recommender._estimator._model._dataset.tag_name_to_idx


# end : class
