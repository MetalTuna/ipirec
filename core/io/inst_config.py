from abc import *
from configparser import ConfigParser
import os


class InstanceConfigParser:

    def __init__(self):
        self._config: ConfigParser = None

    # end : init()

    @abstractmethod
    def make_config(self) -> ConfigParser:
        raise NotImplementedError()

    def conditions_comparision(
        self,
        file_path: str,
    ) -> bool:
        """
        - 요약:
            - 모델의 입출력에서 사용하는 함수입니다.
            - 과거의 분석결과들 중에서, 요청된 실험조건(모델) 결과가 존재하는지 확인하는 기능구현입니다.

        - 매개변수:
            - file_path (str): 과거의 분석결과에 대한 Config 파일의 경로입니다.
                - bin(pickle)과 쌍을 이루는 ini(ConfigParser)를 사용합니다.

        - 반환:
            - bool: 모든 원소가 같다면 참을, 그렇지 않다면 거짓을 반환합니다.
                - 거짓이면 분석처리 후, 결과를 저장하는 기능을 구현하세요.
                    - 현 시점에서는 추상적 구현임
                    - BinConfigPair와 PrevExpDict를 사용한 논리적 구성참조가 필요할 것으로 생각됨.
        """
        if self._config == None:
            return False
        dump_config = InstanceConfigParser.read_config(
            file_path=file_path,
        )

        ## Comparision
        __is_match = True
        for section in self._config.sections():
            if not dump_config.has_section(section):
                return False

            for option in self._config.options(section):
                if not dump_config.has_option(section, option):
                    return False
                # conditions comparision
                __is_match = __is_match and (
                    self._config.get(section, option) == dump_config(section, option)
                )
                if not __is_match:
                    return False
            # end : for (options)
        # end : for (sections)

        return __is_match

    # end : public bool conditions_comparision()

    def load_config(
        self,
        file_path: str,
    ) -> None:
        """
        - 요약:
            - 모델구성을 불러옵니다.
            - 이 함수는 InstanceConfigParser.read_config()를 호출해 처리됩니다.

        - 매개변수:
            - file_path (str): 설정파일 경로를 사용합니다.
        """
        self._config = InstanceConfigParser.read_config(
            file_path=file_path,
        )

    # end : public void load_config()

    @staticmethod
    def read_config(
        file_path: str,
    ) -> ConfigParser:
        """
        - 요약:
            - 모델구성을 불러옵니다.

        - 매개변수:
            - file_path (str): 설정파일 경로를 사용합니다.

        - 예외:
            - FileNotFoundError: 설정파일이 없다면 예외가 발생합니다.

        - 반환:
            ConfigParser: 설정파일에 대한 인스턴스(ConfigParser)를 반환합니다.
        """
        _config = ConfigParser()
        if os.path.exists(file_path) and os.path.isfile():
            _config.read_file(file_path)
        else:
            raise FileNotFoundError()
        return _config

    # end : public static ConfigParse read_config()


# end : class
