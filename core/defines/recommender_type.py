from enum import Enum


class RecommenderType(Enum):
    """
    - 요약:
        - 추천항목 구성을 선택하는 열거자입니다.
    - 구성:
        - ELA로 추천후보목록 생성하고 의사결정 가능성을 구한 후, 이들을 대상으로 추천: E_ELA (1,)
        - 모든 항목에 대한 의사결정 가능성을 구한 후, 이들을 대상으로 추천: E_SCORE (2,)
    """

    E_ELA = (1,)
    """lc_corr.model.ELABasedRecommender"""
    E_SCORE = (2,)
    """lc_corr.model.ScoreBasedRecommender"""


# end : enum
