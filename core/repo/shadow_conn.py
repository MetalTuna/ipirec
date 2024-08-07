from sshtunnel import *
from pymysql import Connection
from pymysql.cursors import DictCursor
from pandas import DataFrame

from ..defines import Machine


class ShadowConnector:
    """
    - 요약:
        - 원시데이터를 가져올 때, 데이터베이스 접근을 관리하기 위한 클래스입니다.
        - 3090에 먼저 접근하고(SSHTunnel), 접근에 실패하면 업무용 노트북의 데이터베이스에 접근합니다.
    - 구성:
        - public DataFrame get_raw_data(REQUEST_QUERY_STRING)만 사용하면 됩니다.
            - 질의문을 요청할 때 연결이 구성되고, 결과를 반환하기 전에 연결구성에 관한 자원들이 해제됩니다.
    """

    def __init__(
        self,
        db_src: Machine = Machine.E_3090,
    ) -> None:
        self.conn: Connection = None
        self.cur: DictCursor = None
        self.tunnel: SSHTunnelForwarder = None
        self.__db_src: Machine = db_src

    # end : private void init()

    def get_raw_data(self, query_str: str) -> DataFrame:
        self.__open__()
        self.cur.execute(query_str)
        try:
            df = DataFrame(self.cur.fetchall())
        except Exception as e:
            print(e)
        finally:
            self.__close__()
        return df

    # end : public DataFrame get_raw_data()

    def __close__(self) -> None:
        if self.tunnel != None:
            if self.tunnel.is_alive:
                self.tunnel.stop()
        if self.cur != None:
            self.cur.close()
        if self.conn != None:
            if self.conn.open:
                self.conn.close()

    # end : private void close()

    def __open__(self) -> None:
        match (self.__db_src):
            case Machine.E_MAC:
                self.__conn_local__()
            case Machine.E_3090:
                self.__conn_3090__()
            case Machine.E_AWS:
                raise NotImplementedError()
        # end : match_case (Machine)

    # end : private void open()

    def __conn_3090__(self) -> None:
        self.tunnel = SSHTunnelForwarder(
            ssh_address_or_host=("210.121.151.231", 23),
            ssh_username="colley",
            ssh_password="colley1212",
            ssh_pkey=None,
            remote_bind_address=("210.121.151.231", 23),
        )
        try:
            self.tunnel.start()
            self.conn = Connection(
                host=self.tunnel.local_bind_host,
                port=3306,
                user="tghwang",
                password="ghkdxorb9999!",
                cursorclass=DictCursor,
            )
            self.conn.connect()
            self.cur: DictCursor = self.conn.cursor()
        except Exception as e:
            print(e)
            self.__close__()
            self.__db_src = Machine.E_MAC
            self.__conn_local__()

    # end : private void conn_3090()

    def __conn_local__(self) -> None:
        try:
            self.conn = Connection(
                host="127.0.0.1",
                port=3306,
                user="taegyu.hwang",
                password="ghkdxorb9999!",
                cursorclass=DictCursor,
            )
            self.conn.connect()
            self.cur: DictCursor = self.conn.cursor()
        except Exception as e:
            print(e)
            self.__close__()
            raise ConnectionError()

    # end : private void conn_local()


# end : class ShadowConnector
