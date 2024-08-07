from enum import Enum


class AnalysisMethodType(Enum):
    """
    - 요약:
        - 의사결정 분석모델을 선택하는 열거자입니다.
    - 구성:
        - 항목 기반 협업 필터링: E_IBCF (1,)
        - 행렬 분해 추천기: E_NMF (2,)
        - 태그 상관관계 기반 추천기: E_IPIRec (3,)
    """

    E_IBCF = (1,)
    """Item-based CF"""
    E_NMF = (2,)
    """NMF-based recommender"""
    E_IPIRec = (3,)
    """IP items analysis"""
