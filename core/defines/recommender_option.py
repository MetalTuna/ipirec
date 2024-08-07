from enum import Enum


class RecommenderOption(Enum):
    """
    - 요약:
        - 추천항목 구성에 대한 설정입니다.
    - 구성:
        - 상위 N개에 속한 항목들을 추천: E_TOP_N (1,)
        - 임계 값 이상인 항목들을 추천, 이들을 대상으로 추천: E_SCORE_THRESHOLD (2,)
    """

    E_TOP_N = (1,)
    """기대점수를 기준으로 상위 N개에 속한 항목들을 추천"""
    E_SCORE_THRESHOLD = (2,)
    """기대점수를 기준으로 임계 값 이상인 항목들을 추천"""

    @staticmethod
    def to_str(selector: Enum) -> str:
        if not isinstance(selector, Enum):
            raise NotImplementedError()
        match (selector):
            case RecommenderOption.E_TOP_N:
                return "TopN"
            case RecommenderOption.E_SCORE_THRESHOLD:
                return "Threshold"
            case _:
                raise NotImplementedError()
        # end : match-case

    # end : public static str to_str()
