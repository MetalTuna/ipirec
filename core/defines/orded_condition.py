from enum import Enum


class OrdedCondition(Enum):
    """
    - 요약:
        - 실험용 데이터 셋의 환경구성에 관한 열거자입니다.
        - 어떤 기준으로 의사결정 데이터 셋을 정렬해서 검증할 것인가를 정의합니다.
    - 정렬조건:
        - E_NOT_DEF: 원시 데이터에서 처리된 나열된 순서로 실험환경을 구성합니다.
        - E_TIMESTAMP: 의사결정한 시간을 기준으로 내림차 순 정렬한 후, 실험환경을 구성합니다.
        - E_RND_SHUFFLE: 무작위로 열거된 순서로 실험환경을 구성합니다.
    """

    E_NOT_DEF = (1,)
    """원시데이터 셋을 그대로 n토막"""
    E_TIMESTAMP = (2,)
    """의사결정한 시간을 기준으로 정렬하고 n토막"""
    E_RND_SHUFFLE = (3,)
    """무작위로 섞고, n토막"""


# end : enum OrdedConditions
