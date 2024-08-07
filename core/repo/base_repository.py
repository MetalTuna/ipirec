import os
from abc import *

from pandas import DataFrame

from ..defines import Machine
from .shadow_conn import ShadowConnector


class BaseRepository(metaclass=ABCMeta):
    """
    - 요약:
        - 항목사전 구축을 위한 추상클래스입니다.
        - 작업은 열거된 순으로 처리됩니다.
            - 원시데이터 복제
            - 원시데이터의 누락 값 처리
            - 도메인 통합에 따른 항목구성
            - 항목사전 출력
    ----
    - 기능:
        - load_data()
            - 원시데이터를 가져옵니다.
                - 데이터베이스의 원시데이터를 가져와 원시파일 생성(optional)
                - 준비된 원시파일을 메모리에 적재
                - 누락 값을 채우거나, 누락 값이 있는 항목을 제거(optional)
        - dump_data()
            - 여러 도메인의 항목들을 하나의 항목으로 통합해서 항목사전을 만들고, 관련정보들을 저장합니다.
                - 각 도메인의 항목과 통합항목의 관계를 저장합니다.
                - 통합항목의 고유번호와 메타데이터를 저장합니다.
        - convert_decision()
            - 도메인이 다른 의사결정 데이터의 고유번호를 통합항목의 고유번호로 변환합니다.
                - 통합항목에 미포함 된 의사결정들은 제거됩니다.

    """

    def __init__(
        self,
        raw_data_path: str,
        db_src: Machine = Machine.E_MAC,
    ) -> None:
        # 폴더 있냐
        if not os.path.exists(raw_data_path):
            os.makedirs(raw_data_path)
        else:
            # 이게 파일이냐
            if os.path.isfile(raw_data_path):
                raise NotADirectoryError()
        self._raw_data_path = raw_data_path
        self.item_dict = dict()
        """
        - 항목의 메타데이터 사전 (dict)
            - Key: item_id (int)
            - Value: item_info (dict)
                - Key: property (str)
                - Value: Values (Any)
        """

        self._connector = ShadowConnector(db_src=db_src)
        """
        - 원시데이터를 가져올 때, 데이터베이스 접근을 관리하는 클래스 인스턴스입니다.
        """

    # end : init()

    def get_raw_data(self, query_str: str) -> DataFrame:
        print("[REQUEST_QUERY_STRING]")
        print(query_str)
        return self._connector.get_raw_data(query_str)

    # end : public DataFrame get_raw_data()

    @abstractmethod
    def load_data(self):
        """원시데이터 가져오기"""
        raise NotImplementedError()

    @abstractmethod
    def dump_data(self):
        """규격에 맞춰 원시데이터를 재단한 복제본 출력"""
        raise NotImplementedError()

    @abstractmethod
    def convert_decision(self):
        """원시데이터 변환"""
        raise NotImplementedError()


# end : class
