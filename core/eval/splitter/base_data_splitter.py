import os
from abc import *

from ...io import DirectoryPathValidator

from core import BaseAction, DecisionType


class BaseDataSplitter(metaclass=ABCMeta):
    """
    - 요약:
        - 실험용 데이터 셋 구성을 위한 추상클래스입니다.
    - 함수:
        - public void split(): 실험용 데이터 셋 구성작업을 실행합니다.
        - protected void preprocess(): 원시데이터 불러오고, 정렬 등의 작업을 진행됩니다.
        - private void dump_validations_set(): 처리된 결과를 설정된 경로에 출력합니다.
    - 추상함수:
        - protected abstract void process(): 실험용 데이터 셋을 구성하는 작업을 구현하세요.
        - protected abstract void postprocess(): 작업결과를 파일로 저장 후, 인스턴스들을 반환하도록 구현하세요.
    """

    def __init__(
        self,
        src_dir_path: str,
        dest_dir_path: str = "",
    ) -> None:
        """
        - 요약: 생성자

        - 매개변수:
            - src_dir_path (str): 실험용 데이터 셋(원본)이 존재하는 폴더 경로입니다.
            - dest_dir_path (str, optional): 실험용 데이터 셋(분리)이 저장될 폴더 경로입니다.
        """
        DirectoryPathValidator.exist_dir(src_dir_path)
        self.SRC_DIR_PATH = src_dir_path
        """원시데이터 셋의 폴더경로"""
        self.DEST_DIR_PATH = src_dir_path if dest_dir_path == "" else dest_dir_path
        """처리결과가 저장될 폴더경로"""
        DirectoryPathValidator.mkdir(self.DEST_DIR_PATH)
        self._raw_decisions_dict: dict = None
        """
        - 요약:
            - 원시 의사결정 데이터 셋에 대한 dictionary
        - 구성:
            - Key: view, like, purchase (str)
            - Value: instance (dict) - BaseAction
        """
        self.validations_decisions_dict: dict = None
        """
        - 요약:
            - 검증용 데이터 집합에 대한 dictionary
        - 구성:
            - Key: 검증집합 번호 (int)
            - Value: "train", "test" (dict) {
                list (BaseAction)
            }
        """
        ## CV, Timestamp (static, dynamic), Decision_type (like, purchase)

    # end : init()

    ## abstract methods
    @abstractmethod
    def _process_(self) -> None:
        """
        - 요약:
            - 실험용 데이터 구성
            - self._raw_decisions_dict 변수로 필요한 작업을 구현하세요.
        """
        raise NotImplementedError()

    @abstractmethod
    def _postprocess_(self) -> None:
        """
        - 요약:
            - 파일로 저장 후, 인스턴스들을 반환
        """
        raise NotImplementedError()

    ## methods
    def split(self) -> None:
        """
        - 요약:
            - 원본 데이터 셋을 조건에 맞게 분리합니다.

        - 매개변수:
            - src_dir_path (str): 실험용 데이터 셋(원본)이 존재하는 폴더 경로입니다.
            - dest_dir_path (str): 실험용 데이터 셋(분리)이 저장될 폴더 경로입니다.
        """
        self._preprocess_()
        self._process_()
        self._postprocess_()
        self.__dump_validations_set__()

    # end : public void split()

    def __dump_validations_set__(self) -> None:
        """
        - 요약:
            - 검증용 데이터 셋을 파일로 저장합니다.
        - 주의:
            - 교차검증 데이터 셋 출력으로 구현됐습니다.
            - 교차검증과 다른 형태로 구성했다면, 그에 맞게 이 함수를 재정의 하세요.
        """
        if self.validations_decisions_dict == None:
            raise NotImplementedError()

        # d_kwd = decisions keywords [view, like, purchase]
        for d_kwd in self.validations_decisions_dict.keys():
            dtype_fold_dict: dict = self.validations_decisions_dict[d_kwd]
            # k = Arg. No. of KFold - [0, 1, ..., K-1]
            for k in dtype_fold_dict.keys():
                validation_set_dict: dict = dtype_fold_dict[k]
                # val_kwd = validation keywords
                for val_kwd in validation_set_dict.keys():
                    decisions_list: list = validation_set_dict[val_kwd]
                    # file_path = ${DEST_PATH} / file_name.csv
                    # file_name = [train, test]_[0, ..., k - 1]_[view, like, purchase]_list, => train_0_view_list.csv
                    file_path = f"{self.DEST_DIR_PATH}/{val_kwd}_{k}_{d_kwd}_list.csv"
                    with open(
                        file=file_path,
                        mode="wt",
                        encoding="utf-8",
                    ) as fout:
                        # csv header
                        fout.write("user_id,item_id,created_time\n")
                        for inst in decisions_list:
                            inst: BaseAction
                            fout.write(
                                f"{inst.user_id},{inst.item_id},{inst.timestamp}\n"
                            )
                        # end : for (decisions)
                        fout.close()
                # end : for (train, test)
            # end : for (KFold)
        # end : for (decision_type)

    # end : private void dump_validations_set()

    def _preprocess_(self) -> None:
        """
        - 요약:
            - 원시데이터 불러오고, 정렬 등의 작업을 진행한다.

        - 예외:
            - FileNotFoundError: 원시 의사결정 파일을 불러오지 못하면 예외가 발생합니다.
        """
        self._raw_decisions_dict = {
            kwd: BaseAction.load_collection(f"{self.SRC_DIR_PATH}/{kwd}_list.csv")
            for kwd in ["view", "like", "purchase"]
        }
        if self._raw_decisions_dict == None:
            raise FileNotFoundError()

    # end : protected void preprocess()


# end : class
