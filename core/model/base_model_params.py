from abc import *


class BaseModelParameters(metaclass=ABCMeta):
    """
    - 요약:
        - 모델을 구성하는 변수들(parameters, arguments)을 설정하는 기능관리를 위한 인터페이스입니다.
        - BaseModel, BaseEstimator가 이 인터페이스를 상속합니다.
            - 분석모델을 구성하는 변수들을 명시적으로 재정의하고, 인스턴스로 모델을 설정하는 기능을 구현하세요.
    - 추상함수:
        - public static BaseModelParameters create_models_parameters(): 여기에서 분석모델의 변수들을 정의합니다.
        - protected void set_model_params(): model_params로 모델을 구성합니다.
    """

    @staticmethod
    def create_models_parameters() -> dict:
        """
        - 요약:
            - 분석모델을 구성하는 변수들을 생성합니다.
            - 생성자에서 모델변수를 명시적으로 선언하고, 이 함수에서 명시적으로 설정하세요.
        - 반환:
            - 모델의 매개변수들을 dictionary로 반환합니다.
        """
        raise NotImplementedError()

    @abstractmethod
    def _set_model_params_(
        self,
        model_params: dict,  # instacne: BaseModelParameters
    ) -> None:
        """
        - 요약:
            - 분석에 사용될 변수들을 설정합니다.

        - 매개변수:
            model_params (dict): create_models_parameters()에서 생성된 dictionary를 사용하세요.
        """
        raise NotImplementedError()


# end : class
