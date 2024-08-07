import os
import sys

WORKSPACE_HOME_PATH = os.path.dirname(__file__)
"""WORKSPACE ROOT PATH STRING"""
sys.path.append(WORKSPACE_HOME_PATH)

from .core import *

# from .lc_corr import *
from .decompositions import *
from .ipirec import *
from .itemcf import *
from .refine import *


__all__ = [
    # [HOME_PATH]
    "WORKSPACE_HOME_PATH",
    # [core]
    "BaseAction",
    "BaseDataSet",
    "BaseEstimator",
    "BaseEvaluator",
    "BaseModel",
    "BaseModelParameters",
    "BaseRecommender",
    "BaseRepository",
    "BaseTrain",
    "DataType",
    "DecisionType",
    "IRMetricsEvaluator",
    "ItemEntity",
    "Machine",
    "ShadowConnector",
    "StatisticsEvaluator",
    "TagEntity",
    "UserEntity",
    # [decomposition]
    "DecompositionModel",
    "NMFDecompositionModel",
    "DecompositionsEstimator",
    # [item_cf]
    "BaseDistanceModel",
    "Pearson",
    "AdjustedWeightedSum",
    # [ipirec]
    "AdjustCorrelationEstimator",
    "BaseCorrelationEstimator",
    "BiasedCorrelationEstimator",
    "CorrelationModel",
    "ColleyAction",
    "ColleyDataSet",
    "ColleyFilteredDataSet",
    "ColleyItemEntity",
    "ELABasedRecommender",
    "ScoreBasedRecommender",
    "MovieLensDataSet",
    "MovieLensFilteredDataSet",
    "MovieLensItemEntity",
    "MovieLensAction",
    # [refine]
    "ColleyRepository",
    "MovieLensRepository",
]
