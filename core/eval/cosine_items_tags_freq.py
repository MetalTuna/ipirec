from .base_rec_items_tags_freq import BaseRecommenderItemsTagsFreq


class CosineItemsTagsFreq(BaseRecommenderItemsTagsFreq):
    """
    - 요약:
        - "추천된 항목들의 태그 출현 빈도 수"가 "검증 데이터에 항목들의 태그 출현 빈도 수"와 일치했는지 코사인으로 비교합니다.
        - `tags_freq_distance()`를 호출해서 결과를 구합니다.
    """

    def __init__(self) -> None:
        super().__init__()

    # end : init()

    def _distance_(self) -> float:
        numer = denom_x = denom_y = 0.0
        for tag_name in self._tags_dict.keys():
            """
            if not tag_name in self._test_tags_freq_dict:
                continue
            x_freq: int = self._test_tags_freq_dict[tag_name]
            y_freq: int = self._rec_tags_freq_dict.get(tag_name, 0)
            """
            x_freq: int = self._test_tags_freq_dict.get(tag_name, 0)
            y_freq: int = self._rec_tags_freq_dict.get(tag_name, 0)

            numer += x_freq * y_freq
            denom_x += x_freq**2
            denom_y += y_freq**2
        # end : for (tags)

        denom_x = (denom_x**0.5) * (denom_y**0.5)
        return 0.0 if denom_x == 0.0 else numer / denom_x

    # end : protected float distance()


# end : class
