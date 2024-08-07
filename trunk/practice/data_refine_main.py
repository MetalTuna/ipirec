import os

from core import BaseRepository, DataType, Machine
from refine import (
    MovieLensRepository,
    ColleyRepository,
    ColleyRepositoryProductItems,
)  # , ColleyRepositoryFilteredTags


class DataSetRefineryMain:
    """
    요약:
        원시데이터를 실험용 데이터 셋으로 구성하는 작업을 구현합니다.
    """

    def __init__(self, selector: DataType) -> None:
        # self.ds_root_path = ("/Users/taegyu.hwang/Documents/tghwang_git_repo/ipitems_analysis/data")
        self.ds_root_path = f"{os.path.dirname(__file__)}/data"

        src_data_home = ""
        repo: BaseRepository = None

        match (selector):
            case DataType.E_MOVIELENS:
                # self.colley()
                src_data_home = f"{self.ds_root_path}/ml"
                repo = MovieLensRepository(src_data_home)
            case DataType.E_COLLEY:
                # self.movielens()
                src_data_home = f"{self.ds_root_path}/colley"
                repo = ColleyRepositoryProductItems(src_data_home)
            case _:
                raise NotImplementedError()
        # end : match-case

        repo.load_data()
        repo.dump_data()
        repo.convert_decision()

    # end : init()


if __name__ == "__main__":
    inst = DataSetRefineryMain(DataType.E_COLLEY)
# end : main()
