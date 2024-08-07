from core.defines.data_type import DataType
from .base_data_splitter import BaseDataSplitter


class DecisionsTypeSplitter(BaseDataSplitter):
    """
    - 요약:
        - 우리의 원시데이터는 다음과 같이 구성됩니다. 이들을 분리해서 분석결과를 평가하려는 클래스입니다.
            - 원시데이터 구성
                - 봤다: 봤다 + 좋다 (optional., + 샀다)
                - 좋다: 좋다
                - 샀다: 샀다

    >>> 의사결정 속성별로 데이터를 재구성합니다.
    """

    def __init__(self) -> None:
        super().__init__()

    # end : init()

    def split(
        self,
        src_dir_path: str,
        dest_dir_path: str,
        data_type: DataType,
    ) -> None:
        ## 원시데이터를 불러온다.
        # 1. 봤다는 봤다만 남긴다.
        # 2. 좋다는 좋다만 남긴다.
        # 3. 샀다는 샀다만 남긴다.
        ## 각 인스턴스들을 파일로 저장한 후, 반환한다.
        raise NotImplementedError()
        return super().split(src_dir_path, dest_dir_path, data_type)


# end : class
