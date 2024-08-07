from enum import Enum


class Machine(Enum):
    """
    요약:
        실행환경을 선택합니다.
    - 각 열거자의 어미는 E(element)로 구성됩니다.
        - E_AWS
        - E_3090
        - E_MAC
    - ELAConfig에서 MACHINE_MAC을 기본 값으로 사용합니다.
    ---
    >>> Machine.`E_AWS`, Machine.`E_3090`, Machine.`E_Mac`


    """

    E_AWS = (0,)
    """Amazone EC2"""
    E_3090 = (1,)
    """Colley-3090"""
    E_MAC = (2,)
    """노트북 (MAC)"""


# end : enum MACHINE_SELECTOR
