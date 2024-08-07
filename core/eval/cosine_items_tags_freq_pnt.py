from .base_rec_items_tags_freq import BaseRecommenderItemsTagsFreq
from ..entity import BaseAction, UserEntity, ItemEntity


class CosineItemsTagsFreqAddPenalty(BaseRecommenderItemsTagsFreq):
    """
    - 요약:
        - "추천된 항목들의 태그 출현 빈도 수"가 "검증 데이터에 항목들의 태그 출현 빈도 수"와 일치했는지 코사인으로 비교합니다.
            - 사용자 단위로 태그 출현 빈도 수의 차를 구합니다.
        - `tags_freq_distance()`를 호출해서 결과를 구합니다.
    """

    def __init__(self) -> None:
        super().__init__()

    # end : init()

    def _aggregation_(self) -> None:
        user_test_tags_dict = dict()
        """
        Key: user_id (int)
        Value: tags_freq (dict) = {
            Key: tag_name (str),
            Value: tag_freq (int),
        }
        """
        user_rec_tags_dict = dict()
        """
        Key: user_id (int)
        Value: tags_freq (dict) = {
            Key: tag_name (str),
            Value: tag_freq (int),
        }
        """

        # [TestSet] Aggr.
        for inst in self._test_set_list:
            inst: BaseAction
            user_id: int = inst.user_id
            item_id: int = inst.item_id
            user: UserEntity = self._user_dict.get(user_id, None)
            if user == None:
                continue
            item: ItemEntity = self._item_dict.get(item_id, None)
            if item == None:
                continue
            if not user_id in user_test_tags_dict:
                user_test_tags_dict.update({user_id: dict()})
            freq_dict: dict = user_test_tags_dict[user_id]
            for tag_name in item.tags_set:
                if not tag_name in freq_dict:
                    freq_dict.update({tag_name: 0})
                freq_dict[tag_name] += 1
            # end : for (T(i))
        # end : for (decisions - test_set)

        # [RecSet] Aggr.
        for user_id in self._user_dict.keys():
            if not user_id in user_rec_tags_dict:
                user_rec_tags_dict.update({user_id: dict()})
            user: UserEntity = self._user_dict[user_id]
            freq_dict: dict = user_rec_tags_dict[user_id]
            for item_id in user.recommended_items_dict.keys():
                item: ItemEntity = self._item_dict[item_id]
                for tag_name in item.tags_set:
                    if not tag_name in freq_dict:
                        freq_dict.update({tag_name: 0})
                    freq_dict[tag_name] += 1
                # end : for (T(i))
            # end : for (items)
        # end : for (users)

        ## ReDef.
        self._test_tags_freq_dict = user_test_tags_dict
        """
        Key: user_id (int)
        Value: tags_freq (dict) = {
            Key: tag_name (str),
            Value: tag_freq (int),
        }
        """
        self._rec_tags_freq_dict = user_rec_tags_dict
        """
        Key: user_id (int)
        Value: tags_freq (dict) = {
            Key: tag_name (str),
            Value: tag_freq (int),
        }
        """

    # end : protected override void aggregation()

    def _distance_(self) -> float:
        numer = denom_x = denom_y = 0.0
        for user_id in self._user_dict.keys():
            act_tags_freq_dict: dict = self._test_tags_freq_dict.get(user_id, dict())
            rec_tags_freq_dict: dict = self._rec_tags_freq_dict.get(user_id, dict())

            # 둘 중 하나라도 채워졌다면 오답에 대한 불이익을 가하도록 분모에 누산
            tags_set: set = set(act_tags_freq_dict.keys()).union(
                set(rec_tags_freq_dict.keys())
            )
            for tag_name in tags_set:
                act_freq: int = act_tags_freq_dict.get(tag_name, 0)
                rec_freq: int = rec_tags_freq_dict.get(tag_name, 0)
                numer += act_freq * rec_freq
                denom_x += act_freq**2
                denom_y += rec_freq**2
            # end : for (tags)
        # end : for (users)

        denom_x = (denom_x**0.5) * (denom_y**0.5)
        return 0.0 if denom_x == 0.0 else numer / denom_x

    # end : protected override float _distance_()


# end : class
