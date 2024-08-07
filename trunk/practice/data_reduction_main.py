# from refine import ColleyRepository, MovieLensRepository

from core import BaseRepository
from refine import ColleyRepositoryFilteredTags, MovieLensRepository
from enum import Enum


class TargetDomain(Enum):
    COLLEY = (0,)
    MOVIELENS = (1,)


class DataSetRefineryMain:
    """
    요약:
        원시데이터를 실험용 데이터 셋으로 구성하는 작업을 구현합니다.
    """

    def __init__(self, selector: TargetDomain) -> None:
        self.ds_root_path = (
            "/Users/taegyu.hwang/Documents/tghwang_git_repo/ipitems_analysis/data"
        )

        repo: BaseRepository = None

        match (selector):
            case TargetDomain.COLLEY:
                # self.colley()
                src_data_home = f"{self.ds_root_path}/colley"
                # repo = ColleyRepository(src_data_home)
                repo = ColleyRepositoryFilteredTags(src_data_home)
            case TargetDomain.MOVIELENS:
                # self.movielens()
                src_data_home = f"{self.ds_root_path}/ml"
                repo = MovieLensRepository(src_data_home)
            case _:
                raise NotImplementedError()
        # end : match-case

        repo.load_data()
        repo.dump_data()
        repo.convert_decision()

    # end : init()

    """
    def movielens(self) -> None:
        # IP를 가져올 수 없는 영화 수가 약 2000개(9742->7977)
        src_data_home = f"{self.ds_root_path}/ml"
        repo = MovieLensRepository(src_data_home)
        repo.load_data()
        repo.dump_data()
        repo.convert_decision()

    def colley(self) -> None:
        src_data_home = f"{self.ds_root_path}/colley"
        # repo = ColleyRepository(src_data_home)
        repo = ColleyRepositoryFilteredTags(src_data_home)
        repo.load_data()
        repo.dump_data()
        repo.convert_decision()
    """


# end : class

if __name__ == "__main__":
    inst = DataSetRefineryMain(TargetDomain.COLLEY)
    # inst = Main(TargetDomain.MOVIELENS)
    print()
# end : main()
