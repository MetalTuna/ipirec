from ..defines import *
from ..io import DirectoryPathValidator


class Benchmark:

    def __init__(
        self,
        dataset_home_path: str = "",
    ) -> None:
        self.__VALIDATION_OPT: ValidationType = None
        self.__validation_file_path_dict = {
            {"train": list()},
            {"test": list()},
        }
        """
        - Key: ["train", "test"]
        - Value: file_path_list list()
        """
        self.__selected_dataset: DataType = None
        """Colley vs. MovieLens ?"""

        workspace_home = DirectoryPathValidator.get_workspace_path("ipirec")
        dataset_home = ""
        self.__RAW_DATASET_HOME_PATH = (
            dataset_home_path if dataset_home_path != "" else f"{workspace_home}/data"
        )
        """ [Default] `${WORKSPACE_HOME}/data` """
        if not DirectoryPathValidator.exist_dir(self.__RAW_DATASET_HOME_PATH):
            raise FileNotFoundError()

    # end : init()

    def __eval_env_def__(
        self,
        validation_type: ValidationType = ValidationType.E_NONE,
        **kwargs,
    ) -> None:
        """
        - 요약:
            - 검증 조건에 대한 실험환경들을 구성합니다.

        - 매개변수:
            - validation_type (ValidationType, optional): 검증조건을 선택하세요.
            - kwargs (dict): 검증조건에 대한 변수들의 dictionary
        """
        ## selector value validation;
        match (validation_type):
            case ValidationType.E_NONE:
                raise NotImplementedError()
            case ValidationType.E_KFOLD:
                raise NotImplementedError()
            case _:
                raise ValueError()
        # end : match-case
        self.__VALIDATION_OPT = validation_type

        ## get args and parsing
        ## dataset redefine
        # raw_dataset IO => build validation-set (train-test pair)
        # assign Inst.
        ## build eval models
        ## analysis
        ## evals aggregation
        # each VAL_TYPE
        ## results summary
        # dump
        # figures drawing

    # end : private void eval_env_def()

    def __build_validation_dataset__(self) -> None:
        train_file_path = ""
        test_file_path = ""
        file_name_format = ""

        match (self.__VALIDATION_OPT):
            case ValidationType.E_NONE:
                # raw dataset
                # train
                # test
                self.__validation_file_path_dict

                for type_inst in DecisionType:
                    kwd = DecisionType.to_str(type_inst)
                    file_path = ""
                raise NotImplementedError()
            case ValidationType.E_KFOLD:
                # fold_dataset
                raise NotImplementedError()


# end : class


if __name__ == "__main__":
    """
    inst = Benchmark()
    selected_dataset = DataType.E_MOVIELENS
    selected_validation = ValidationType.E_KFOLD
    # val_args
    selected_model = AnalysisMethodType.E_NMF
    # model_args
    selected_metrics = MetricType.E_RETRIEVAL

    for ds_type in DataType:
        for val_type in ValidationType:
            for model_type in AnalysisMethodType:
                for metric_type in MetricType:
                    ## preprocess() - build eval dataset, model params def
                    ## process() - Benchmark.run()
                    ## postprocess() - aggr., summ., fig.
                    raise NotImplementedError()
                # end : for (IR, Stats.)
            # end : for (IBCF, NMF, IPA)
        # end : for (None, KFold)
    # end : for (ML, Colly)
    """
# end : if
