from enum import Enum


class DecisionType(Enum):
    """
    - 요약:
        - 의사결정 종류를 선택하는 열거자입니다.
    - 구성:
        - 봤다: E_VIEW (1,)
        - 좋다: E_LIKE (2,)
        - 샀다: E_PURCHASE (3,)
    """

    E_VIEW = (1,)
    """봤다"""
    E_LIKE = (2,)
    """좋아요"""
    E_PURCHASE = (3,)
    """구매했다(상품구매,게시글등록,판매요청으로 구성)."""

    @staticmethod
    def from_str(decision_str: str):
        inst: DecisionType = None
        if decision_str == "view":
            inst = DecisionType.E_VIEW
        if decision_str == "like":
            inst = DecisionType.E_LIKE
        if decision_str == "purchase":
            inst = DecisionType.E_PURCHASE
        if inst == None:
            raise ValueError()
        return inst

    # end : public static DecisionType from_str()

    @staticmethod
    def to_str_list() -> list:
        return [DecisionType.to_str(decision_type) for decision_type in DecisionType]

    @staticmethod
    def list_to_kwd_str(decision_types_list: list) -> str:
        _h_kwd_str = ""
        for dtype in decision_types_list:
            _h_kwd_str += DecisionType.to_str(dtype)[0]
        return _h_kwd_str

    @staticmethod
    def to_str(selected) -> str:
        if not isinstance(selected, DecisionType):
            raise ValueError()
        match (selected):
            case DecisionType.E_VIEW:
                return "view"
            case DecisionType.E_LIKE:
                return "like"
            case DecisionType.E_PURCHASE:
                return "purchase"
            case _:
                raise ValueError()
        # end : match-case

    # end : public static str to_str()


# end : enum DecisionType
