import os
import sys
import pickle

WORKSPACE_HOME = os.path.dirname(__file__).replace("/experiments", "")
sys.path.append(WORKSPACE_HOME)

from core import *
from lc_corr import *
from itemcf import *

# from decompositions import NMFDecompositionModel, DecompositionsEstimator

if __name__ == "__main__":
    dump_dir_path = f"{WORKSPACE_HOME}/experiments/similarities"
    # 결과 값이 같지만, 동시접근 문제발생이 우려돼 분리
    similarity_file_path = f"{dump_dir_path}/item_pcc_ml_purchase.bin"
    dataset_dir_path = f"{WORKSPACE_HOME}/data/ml"
    testset_file_path = f"{dataset_dir_path}/purchase_list.csv"
    results_summary_path = (
        f"{WORKSPACE_HOME}/results/item_cf_ml_TopN_purchase_IRMetrics.csv"
    )

    # 모델의 매개변수들
    top_n_conditions = [n for n in range(3, 21, 2)]

    # 데이터 셋 불러오기
    dataset = MovieLensFilteredDataSet(dataset_dir_path=dataset_dir_path)
    dataset.load_dataset()

    # 모델 구성하기
    model: Pearson = None
    if os.path.exists(similarity_file_path):
        with open(
            file=similarity_file_path,
            mode="rb",
        ) as fin:
            model: Pearson = pickle.load(fin)
            fin.close()
        # end : StreamReader()
    else:
        model = Pearson(dataset=dataset)
        model.analysis()
        DirectoryPathValidator.mkdir(os.path.dirname(similarity_file_path))
        with open(file=similarity_file_path, mode="wb") as fout:
            pickle.dump(model, file=fout)
            fout.close()
        # end : StreamWriter()

    # 학습하기
    estimator = AdjustedWeightedSum(model=model)

    # 예측 점수를 기준으로 추천하기
    recommender = ScoreBasedRecommender(estimator=estimator)
    file_path = f"{WORKSPACE_HOME}/results/predicted_scores_ml.bin"
    # contains_recommender_binary_file?
    if os.path.exists(file_path):
        with open(file=file_path, mode="rb") as fin:
            recommender: ScoreBasedRecommender = pickle.load(fin)
            fin.close()
        # end : StreamReader()
    else:
        recommender.prediction()

    if not os.path.exists(file_path):
        with open(file=file_path, mode="wb") as fout:
            pickle.dump(recommender, fout)
            fout.close()
        # end : StreamWriter()

    # 성능평가하기
    evaluator = IRMetricsEvaluator(
        recommender=recommender,
        file_path=testset_file_path,
    )
    evaluator.top_n_eval(top_n_conditions=top_n_conditions)
    evaluator.evlautions_summary_df().to_csv(path_or_buf=results_summary_path)

# end : main()
