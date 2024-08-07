from core.defines.data_type import DataType
from .base_data_splitter import BaseDataSplitter


class TimeSeriesSplitter(BaseDataSplitter):
    """
    - 요약:
        - 원시데이터의 의사결정들을 모두 합친 후, 의사결정한 시간을 기준으로 실험용 데이터 셋을 구성합니다.
    """

    def __init__(self) -> None:
        super().__init__()

    def split(
        self,
        src_dir_path: str,
        dest_dir_path: str,
        data_type: DataType,
    ) -> None:
        ## 의사결정별로 분리된 원시데이터를 불러온다.
        ## 시간을 기준으로 분리한다.
        # 1. 정적인 시간 간격으로
        # 2. 동적인 시간 간격으로
        ## 합쳐진 의사결정 데이터를 다시 분리한다.
        ## 각 인스턴스들을 파일로 저장한 후, 반환한다.
        raise NotImplementedError()
        return super().split(src_dir_path, dest_dir_path, data_type)

    # end : public override void split()


# end : class
