import os
from enum import Enum


class DataType(Enum):
    """
    - 요약:
        - 데이터 종류를 선택하는 열거자입니다.
    - 구성:
        - 영화데이터: E_MOVIELENS (1,)
        - 우리데이터: E_COLLEY (2,)
    """

    E_MOVIELENS = (1,)
    """영화데이터"""
    E_COLLEY = (2,)
    """우리데이터"""

    @staticmethod
    def dir_path_str_to_inst(dataset_dir_path: str):
        _dirname = os.path.dirname(dataset_dir_path)
        type_str = os.path.basename(_dirname)

        if type_str == "colley":
            return DataType.E_COLLEY
        elif type_str == "ml":
            return DataType.E_MOVIELENS
        else:
            raise ValueError()

    @staticmethod
    def to_str(selected) -> str:
        if not isinstance(selected, DataType):
            raise ValueError()

        match (selected):
            case DataType.E_COLLEY:
                return "colley"
            case DataType.E_MOVIELENS:
                return "ml"
            case _:
                raise ValueError()

    # end : public static string to_str()


# end : enum DataType
