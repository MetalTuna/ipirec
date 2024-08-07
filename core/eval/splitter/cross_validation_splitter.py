from sklearn.model_selection import KFold

from core import DataType
from .base_data_splitter import BaseDataSplitter


class CrossValidationSplitter(BaseDataSplitter):
    """
    - 요약:
        - 원시데이터의 의사결정을 n토막 내서 실험용 데이터 셋을 구성합니다.

    >>> if (n == 5) then,
    >>> view[1, 2, 3, 4, 5] => 0: view_train = [1, 2, 3, 4]; view_test = [5]
    """

    def __init__(
        self,
        src_dir_path: str,
        dest_dir_path: str = "",
        fold_k: int = 5,
        orded_timestamp: bool = True,
    ) -> None:
        super().__init__(src_dir_path, dest_dir_path)
        self.__orded_timestamp = orded_timestamp
        """시간으로 정렬"""
        self.__fold = KFold(fold_k)
        """쪼개기"""

    # end : init()

    def _process_(self) -> None:
        ## 필요하다면 원시데이터 정렬
        if self.__orded_timestamp:
            for kwd in self._raw_decisions_dict.keys():
                orded_decisions: list = self._raw_decisions_dict[kwd]
                self._raw_decisions_dict[kwd] = sorted(
                    orded_decisions, key=lambda x: x.timestamp, reverse=True
                )
            # end : for (decisions_type)
        # end : if

        self.validations_decisions_dict = dict()
        for kwd in self._raw_decisions_dict.keys():
            decisions_dict = dict()
            orded_decisions: list = self._raw_decisions_dict[kwd]
            for k, (train_idx, test_idx) in enumerate(
                self.__fold.split(orded_decisions)
            ):
                validation_set_dict = dict()
                train_list = [orded_decisions[idx] for idx in train_idx]
                test_list = [orded_decisions[idx] for idx in test_idx]
                validation_set_dict.update({"train": train_list})
                validation_set_dict.update({"test": test_list})
                # self.validations_decisions_dict.update({k: validation_set_dict})
                decisions_dict.update({k: validation_set_dict})
            # end : for (KFold)
            self.validations_decisions_dict.update({kwd: decisions_dict})
        # end : for (decisions_type)

    # end : protected override void process()

    def _postprocess_(self) -> None:
        pass

    # end : protected override void postprocess()


# end : class
