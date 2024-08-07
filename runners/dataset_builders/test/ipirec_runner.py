## build in
import os
import sys

# [DEF] Environment variables
__dir_name__ = os.path.basename(os.path.dirname(__file__))
WORKSPACE_HOME = os.path.dirname(__file__).replace(
    f"/runners/dataset_builders/{__dir_name__}", ""
)
""".../ipirec"""
DATA_SET_HOME = f"{WORKSPACE_HOME}/data"
""">>> `${WORKSPACE_HOME}`/data"""
RESULTS_SUMMARY_HOME = f"{WORKSPACE_HOME}/results/IPIRec"
""">>> `${WORKSPACE_HOME}`/results/IPIRec"""
MODEL_NAME = "IPIRec"

# [SET] Env.append($WORKSPACE_HOME)
sys.path.append(WORKSPACE_HOME)

## Custom LIB.
from core import *
from colley import *
from ipirec import *
from movielens import *
from .base_rec_runner import BaseRecommendRunner


class IPIRecommendRunner(BaseRecommendRunner):
    def __init__(
        self,
        temp_dir_path: str = "",
    ) -> None:
        super().__init__(temp_dir_path)

    # end : init()

    def _build_model_(
        self,
        top_n_tags: int = 10,
        co_occur_items_threshold: int = 4,
    ) -> None:
        self.model_params = CorrelationModel.create_models_parameters(
            top_n_tags=top_n_tags,
            co_occur_items_threshold=co_occur_items_threshold,
        )
        self.model = CorrelationModel(
            dataset=self.dataset,
            model_params=self.model_params,
        )
        self.model.analysis()

    # end : protected override void build_model()

    def _build_estimator_(
        self,
        selected_estimator: EstimatorType,
        score_iterations: int,
        score_learning_rate: float,
        score_generalization: float,
        weight_iterations: int,
        weight_learning_rate: float,
        weight_generalization: float,
        train_decisions_sequence_list: list = [
            DecisionType.E_VIEW,
            DecisionType.E_LIKE,
            DecisionType.E_PURCHASE,
        ],
        frob_norm: int = 1,
        default_voting: float = 0.0,
    ) -> None:
        match (selected_estimator):
            case EstimatorType.E_IPIREC_BASE_ESTIMATOR:
                self.estimator_params = (
                    BaseCorrelationEstimator.create_models_parameters(
                        score_iterations=score_iterations,
                        score_learning_rate=score_learning_rate,
                        score_generalization=score_generalization,
                        weight_iterations=weight_iterations,
                        weight_learning_rate=weight_learning_rate,
                        weight_generalization=weight_generalization,
                        frob_norm=frob_norm,
                        default_voting=default_voting,
                    )
                )
                self.estimator = BaseCorrelationEstimator(
                    model=self.model,
                    model_params=self.estimator_params,
                )
            case EstimatorType.E_IPIREC_BIAS_ESTIMATOR:
                self.estimator_params = (
                    BiasedCorrelationEstimator.create_models_parameters(
                        score_iterations=score_iterations,
                        score_learning_rate=score_learning_rate,
                        score_generalization=score_generalization,
                        weight_iterations=weight_iterations,
                        weight_learning_rate=weight_learning_rate,
                        weight_generalization=weight_generalization,
                        frob_norm=frob_norm,
                        default_voting=default_voting,
                    )
                )
                self.estimator = BiasedCorrelationEstimator(
                    model=self.model,
                    model_params=self.estimator_params,
                )
            case _:
                raise ValueError()
        # end : match-case (Estimator)

        self.estimator.adjust_tags_score()
        # self.estimator._adjust_tags_corr_()
        self._train_decisions_sequence_list = train_decisions_sequence_list
        for decision_type in self._train_decisions_sequence_list:
            self.estimator.train(
                target_decision=decision_type,
                n=weight_iterations,
            )
        # end : for (decision_types)

    # end : protected override void build_estimator()


# end : class
