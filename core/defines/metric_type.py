from enum import Enum


class MetricType(Enum):
    """
    - 요약:
        - 추천결과에 대한 모델의 성능평가 방법을 선택하는 열거자입니다.
    - 구성:
        - 정보 검색에서 사용되는 평가척도: E_RETRIEVAL (1,)
        - 통계이론에서 사용되는 평가척도: E_STATISTICS (2,)
    """

    E_RETRIEVAL = (1,)
    """information retrieval measurement and metrics"""
    E_STATISTICS = (2,)
    """statistical distributions measurements"""
