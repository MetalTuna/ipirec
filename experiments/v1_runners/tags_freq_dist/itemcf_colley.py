###
# 24.05.23 12:04 Observ. Corr. Res.
###


### buildt-in
import os
import sys
import pickle

### 3rd Pty.
import numpy as np

### Custom LIB
__CURRRENT_DIR_PATH = os.path.dirname(__file__)
WORKSPACE_HOME = __CURRRENT_DIR_PATH.replace(
    f"/experiments/{os.path.basename(__CURRRENT_DIR_PATH)}", ""
)
sys.path.append(WORKSPACE_HOME)
# print(WORKSPACE_HOME)
# exit()

from core import *
from colley import *
from itemcf import *
from ipirec import *
from movielens import *
from rec_tags_freq import *

if __name__ == "__main__":
    selected_data: DataType = None
    selected_decision: DecisionType = None
    dataset: BaseDataSet = None
    model: BaseModel = None
    estimator: BaseEstimator = None
    recommender: BaseRecommender = None
    evaluator: BaseEvaluator = None

    model_str = "ItemCF"
    selected_data = DataType.E_COLLEY
    # selected_decision = DecisionType.E_LIKE
    selected_decision = DecisionType.E_PURCHASE

    KFold = 5

    top_n_conditions = [n for n in range(3, 37, 2)]

    dataset_str = DataType.to_str(selected_data)
    decision_str = DecisionType.to_str(selected_decision)
    dataset_dir_path = f"{WORKSPACE_HOME}/data/{dataset_str}"
    # testset_file_path = f"{dataset_dir_path}/{decision_str}_list.csv"
    results_dir_path = f"{WORKSPACE_HOME}/results"

    opt_file_name_str = f"{model_str}_{dataset_str}"
    results_summary_path = f"{results_dir_path}/{opt_file_name_str}_TopN_IRMetrics.csv"

    tags_freq_dist_dict = dict()
    """
    Key: set_id (int)
    Value: distance (float)
    """

    ### figures_file_name_sty: ${RESULTS_HOME_PATH}/${MODEL_OPT}_${ELEMENTS}_${SET_NO}_${DECISION} ###
    """
    raw_corr_figure_file_path = (
        f"{results_dir_path}/{opt_file_name_str}_raw_tags_corr_{k}_{decision_str}.svg"
    )
    biased_corr_figure_file_path = f"{results_dir_path}/{opt_file_name_str}_biased_tags_corr_{k}_{decision_str}.svg"
    corr_weight_figure_file_path = f"{results_dir_path}/{opt_file_name_str}_tags_corr_wegiht_{k}_{decision_str}.svg"
    adjusted_corr_figure_file_path = f"{results_dir_path}/{opt_file_name_str}_adjuted_tags_corr_{k}_{decision_str}.svg"
    """

    for k in range(KFold):
        match (selected_data):
            case DataType.E_COLLEY:
                dataset = ColleyFilteredDataSet(dataset_dir_path)
            case DataType.E_MOVIELENS:
                dataset = MovieLensFilteredDataSet(dataset_dir_path)
            case _:
                raise NotImplementedError()
        # end : match-case

        dataset._load_metadata_()
        for decision_type in DecisionType:
            file_path = str.format(
                "{0}/train_{1}_{2}_list.csv",
                dataset_dir_path,
                k,
                DecisionType.to_str(decision_type),
            )
            dataset.append_decisions(
                file_path=file_path,
                decision_type=decision_type,
            )
        # end : for (DecisionTypes)
        dataset.__id_index_mapping__()

        file_path = str.format(
            "{0}/resources/similarities/{1}_{2}_item_pcc.bin",
            WORKSPACE_HOME,
            dataset_str,
            k,
        )
        if os.path.exists(file_path):
            with open(file=file_path, mode="rb") as fin:
                model: Pearson = pickle.load(file=fin)
                fin.close()
        else:
            model = Pearson(dataset=dataset)
            model.analysis()
            with open(file=file_path, mode="wb") as fout:
                pickle.dump(obj=model, file=fout)
                fout.close()
        estimator = AdjustedWeightedSum(model=model)

        # 예측 점수를 기준으로 추천하기
        recommender = ScoreBasedRecommender(estimator=estimator)
        recommender.prediction()

        # 성능평가하기
        file_path = str.format(
            "{0}/test_{1}_{2}_list.csv",
            dataset_dir_path,
            k,
            decision_str,
        )
        evaluator = IRMetricsEvaluator(
            recommender=recommender,
            file_path=file_path,
        )
        evaluator.top_n_eval(top_n_conditions=top_n_conditions)
        evaluator.evlautions_summary_df().to_csv(path_or_buf=results_summary_path)

        # tags_freq_dist = CosineItemsTagsFreq()
        tags_freq_dist = CosineItemsTagsFreqAddPenalty()
        distance: float = tags_freq_dist.tags_freq_distance(
            test_set=evaluator.TEST_SET_LIST,
            recommender=recommender,
        )
        tags_freq_dist_dict.update({k: distance})
        print(f"tags freq. distance (cosine) [{k}/{KFold}]: {distance}")

    # end : for (KFold)

    numer = 0.0
    for k, distance in tags_freq_dist_dict.items():
        numer += distance
        print(f"Set {k}: {distance}\n")
    # end : for (KFold)
    numer = numer / len(tags_freq_dist_dict)
    print(f"Mean: {numer}")
# end : main()
