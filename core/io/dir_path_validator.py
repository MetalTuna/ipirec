import os
from datetime import datetime


class DirectoryPathValidator:
    """
    - 요약:
        - 폴더경로 관리를 위한 클래스입니다.
    """

    @staticmethod
    def current_datetime_str() -> str:
        """
        - 요약:
            - 이 함수가 호출된 시간을 반환합니다. (YYYYMMDD_HHMMSS)
                - ex. 20240504_161103
        """
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @staticmethod
    def get_workspace_path(workspace_name: str) -> str:
        """
        - 요약:
            - 작업경로를 불러옵니다.

        - 매개변수:
            - workspace_name (str): 작업명

        - 예외:
            - FileNotFoundError: 작업명과 다르면 경로를 구할 수 없습니다.

        Returns:
            str: 작업경로
        """
        depth_dir_list = os.path.dirname(__file__).split("/")
        if not workspace_name in depth_dir_list:
            raise FileNotFoundError()
        cur_dir_path = ""
        for dir_name in depth_dir_list:
            cur_dir_path += f"/{dir_name}"
            if dir_name == workspace_name:
                break
        # end : for (depth_dir_list)

        return cur_dir_path

    # end : public static str get_workspace_path()

    @staticmethod
    def exist_dir(dir_path: str) -> bool:
        """
        - 요약:
            - 해당경로에 폴더가 존재하는지 확인합니다.

        - 매개변수:
            - dir_path (str): 확인할 폴더경로

        - 예외:
            - NotADirectoryError: 동명의 파일이 있다면 예외가 발생합니다.

        - 반환:
            - bool: 폴더있냐
        """
        if os.path.isfile(dir_path):
            print(dir_path)
            raise NotADirectoryError()
        return os.path.exists(dir_path)

    # end : public static bool exist_dir()

    @staticmethod
    def mkdir(dir_path: str) -> None:
        """
        - 요약:
            - 폴더만들기

        - 매개변수:
            - dir_path (str): 대상 폴더경로
        """

        # 그러면 재귀적으로 확인하면 되겠네
        mkdir_list = list()
        if DirectoryPathValidator.exist_dir(dir_path):
            return
        mkdir_list.append(dir_path)
        while dir_path != "":
            dir_name_str = f"/{os.path.basename(dir_path)}"
            dir_path = dir_path[: len(dir_path) - len(dir_name_str)]
            if DirectoryPathValidator.exist_dir(dir_path):
                break
            mkdir_list.append(dir_path)
        # end : while ()
        mkdir_list = sorted(
            mkdir_list,
            key=lambda x: len(x),
            reverse=False,
        )
        for dir_str in mkdir_list:
            os.mkdir(dir_str)
        # end : for (mkdir_paths)

    # end : public static bool mkdir()


# end : class
