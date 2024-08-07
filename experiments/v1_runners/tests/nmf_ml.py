import os
import sys

WORKSPACE_HOME = os.path.dirname(__file__).replace("/experiments", "")
sys.path.append(WORKSPACE_HOME)
# print(WORKSPACE_HOME)
# exit()


class NMFExperiments:
    def run(self) -> None:
        dataset_dir_path = f"{WORKSPACE_HOME}/data/ml"
        splitter = CrossValidationSplitter(src_dir_path=dataset_dir_path)
        splitter.split()

    # end : public void run()


# end : class

from core import *
from lc_corr import *
from decompositions import NMFDecompositionModel, DecompositionsEstimator

if __name__ == "__main__":
    dataset_dir_path = f"{WORKSPACE_HOME}/data/ml"
    # testset_file_path = f"{dataset_dir_path}/like_list.csv"
    testset_file_path = f"{dataset_dir_path}/purchase_list.csv"
    results_summary_path = f"{WORKSPACE_HOME}/results/nmf_ml_TopN_IRMetrics.csv"

    # 모델의 매개변수들
    ## 제자리....
    factors_dim = 200  #
    iterations = 50  #
    dist_n = 1
    learning_rate = 0.00002  #
    generalization = 0.000002  # *= 10
    top_n_conditions = [n for n in range(3, 21, 2)]

    # 데이터 셋 불러오기
    dataset = MovieLensFilteredDataSet(dataset_dir_path=dataset_dir_path)
    dataset.load_dataset()

    # 모델 구성하기
    model_params = NMFDecompositionModel.create_models_parameters(
        factors_dim=factors_dim,
    )
    model = NMFDecompositionModel(
        dataset=dataset,
        model_params=model_params,
    )
    model.analysis()

    # 학습하기
    model_params = DecompositionsEstimator.create_models_parameters(
        learning_rate=learning_rate,
        generalization=generalization,
    )
    estimator = DecompositionsEstimator(
        model=model,
        model_params=model_params,
    )
    estimator.train(DecisionType.E_VIEW, n=dist_n, emit_iter_condition=iterations)
    estimator.train(DecisionType.E_LIKE, n=dist_n, emit_iter_condition=iterations)
    estimator.train(DecisionType.E_PURCHASE, n=dist_n, emit_iter_condition=iterations)

    # 예측 점수를 기준으로 추천하기
    recommender = ScoreBasedRecommender(estimator=estimator)
    recommender.prediction()

    # 성능평가하기
    evaluator = IRMetricsEvaluator(
        recommender=recommender,
        file_path=testset_file_path,
    )
    evaluator.top_n_eval(top_n_conditions=top_n_conditions)
    evaluator.evlautions_summary_df().to_csv(path_or_buf=results_summary_path)

# end : main()
