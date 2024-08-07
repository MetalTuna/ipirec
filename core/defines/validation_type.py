from enum import Enum


class ValidationType(Enum):
    """
    - 요약:
        - 검증방법의 명세입니다.
    - 구성:
        - 설정없음: E_NONE (1,)
        - K회 교차검증: E_KFOLD (2,)
    """

    E_NONE = (1,)
    """
    - 요약:
        - 학습/검증 집합 구분 기준이 없습니다.
        - 원시 데이터 셋을 그대로 사용합니다.
    """
    E_KFOLD = (2,)
    """
    - 요약:
        - K회 교차검증으로 구성합니다.
    """


# end : enum
