"""
- 요약:
    - 데이터 분석을 다루는 추상클래스들로 구성된 모듈입니다.
    - 정의된 구성에 맞게, 각 클래스들을 상속해서 구현하세요.
    
- 구성:
    - BaseDataSet: 데이터의 입출력을 추상화 합니다.
    - BaseModelParameters: 분석모델에 사용되는 변수들을 명세합니다.
        - BaseModel, BaseEstimator가 이 인터페이스를 상속합니다.
    - BaseModel: 입력된 데이터의 전처리, 특성 및 특징추출, 상관관계 등을 추상화 합니다.
    - BaseEstimator: BaseModel을 통해 분석된 특성들을 사용해 추정하는 기능을 추상화 합니다.
        - BaseTrain: 추정(예측)오차를 모델에 피드백하는 기능을 추상화 합니다. 이 클래스는 BaseEstimator를 상속합니다.
    - BaseRecommender: 모델의 추천기능을 추상화 합니다.
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

from .base_dataset import BaseDataSet
from .base_model import BaseModel
from .base_estimator import BaseEstimator
from .base_train import BaseTrain
from .base_recommender import BaseRecommender
from .base_model_params import BaseModelParameters

## objective function modules
from .base_objective_score import BaseObjectiveScore
from .base_objective_train import BaseObjectiveTrain
