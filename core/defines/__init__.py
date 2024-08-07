"""
- 요약:
    - 정의된 열거자들을 구성하는 모듈입니다.
    - 여기에 정의된 열거자들의 원소들은 어미에 E_로 작성됐습니다.
    
- 구성:
    - DecisionType: 의사결정의 종류를 정의합니다. 봤다(E_VIEW), 좋다(E_LIKE), 샀다(E_PURCHASE).
    - DataType: 데이터의 종류를 정의합니다. 영화데이터(E_MOVIELENS), 우리데이터(E_COLLEY).
    - Machine: 접근할 데이터 베이스를 정의합니다. 아마존(E_AWS), 3090(E_3090), 업무용 노트북(E_MAC).
    - ValidationType: 검증방법의 명세입니다. 설정없음(E_NONE), 교차검증(E_KFOLD)
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

## TEST
from .tag_similarity_type import TagSimilarityType

## FIXED
from .decision_type import DecisionType
from .data_type import DataType
from .machine import Machine
from .validation_type import ValidationType
from .analysis_method_type import AnalysisMethodType
from .metric_type import MetricType
from .recommender_type import RecommenderType
from .recommender_option import RecommenderOption
from .estimator_type import EstimatorType
