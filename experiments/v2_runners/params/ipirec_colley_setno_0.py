"""
[24.05.06] IPIRec - MODEL PARAMS: heuristics approach;
- CV Set No. 1;
"""

## build-in
import os
import sys

# [DEF] Environment variables
__dir_name__ = os.path.basename(os.path.dirname(__file__))
WORKSPACE_HOME = os.path.dirname(__file__).replace(f"/experiments/{__dir_name__}", "")
""".../ipirec"""
DATA_SET_HOME = f"{WORKSPACE_HOME}/data"
""">>> `${WORKSPACE_HOME}`/data"""
RESULTS_SUMMARY_HOME = f"{WORKSPACE_HOME}/results/params"
""">>> `${WORKSPACE_HOME}`/results/params"""
MODEL_NAME = "NMF"

# [SET] Env.append($WORKSPACE_HOME)
sys.path.append(WORKSPACE_HOME)

## custom modules
from core import *
from lc_corr import *
from decompositions import *


def tags_freq_aggr(dataset: BaseDataSet) -> None:
    numer = denom = 0.0
    for user_id in dataset.user_dict.keys():
        user: UserEntity = dataset.user_dict[user_id]
        tags_cnt = len(user.tags_decision_freq_dict)
        if tags_cnt == 0:
            continue
        numer += tags_cnt
        denom += 1
        # end : for (T(u))
    # end : for (users)
    numer = numer / denom
    print(numer)


# end : public void tags_freq_aggr()


if __name__ == "__main__":
    selected_data = DataType.E_COLLEY
    set_no = 0
    top_n_items_list = [n for n in range(5, 27, 2)]
    dataset_dir_path = f"{DATA_SET_HOME}/{DataType.to_str(selected_data)}"
    dataset = ColleyFilteredDataSet(dataset_dir_path=dataset_dir_path)
    dataset._load_metadata_()
    for decision_type in DecisionType:
        kwd = DecisionType.to_str(decision_type)
        training_file_path = f"{dataset_dir_path}/train_{set_no}_{kwd}_list.csv"
        dataset.append_decisions(
            file_path=training_file_path,
            decision_type=decision_type,
        )
    # end : for (decisions)
    dataset.__id_index_mapping__()

    # tags_freq_aggr(dataset=dataset)
    # \\mu(T) ~3.5
    # 2024.05.06 TP = 1205
    frob_norm_n = 2
    top_n_tags = 4
    default_voting = 0.0
    co_occur_items = 5
    iter = 50  # 20
    learning_rate = 0.1
    generalization = 0.05

    model_params = CorrelationModel.create_models_parameters(
        top_n_tags=top_n_tags,
        co_occur_items_threshold=co_occur_items,
        iterations_threshold=iter,
        learning_rate=learning_rate,
    )
    model = CorrelationModel(
        dataset=dataset,
        model_params=model_params,
    )
    model.analysis()
    model_params = BiasedEstimator.create_models_parameters(
        default_voting_score=default_voting,
        learning_rate=learning_rate,
        generalization=generalization,
    )
    estimator = BiasedEstimator(
        model=model,
        model_params=model_params,
    )
    for decision_type in DecisionType:
        estimator.train(
            target_decision=decision_type,
            n=frob_norm_n,
            emit_iter_condition=iter,
        )
    # end : for (decisions)

    recommender = ScoreBasedRecommender(
        estimator=estimator,
    )
    recommender.prediction()

    for test_decision in [DecisionType.E_LIKE, DecisionType.E_PURCHASE]:
        kwd = DecisionType.to_str(test_decision)
        test_file_path = f"{dataset_dir_path}/test_{set_no}_{kwd}_list.csv"
        evaluator = IRMetricsEvaluator(
            recommender=recommender,
            file_path=test_file_path,
        )
        evaluator.top_n_eval(top_n_conditions=top_n_items_list)
        results_df = evaluator.evlautions_summary_df()
        print(f"[{kwd}]\n")
        print(results_df)
# end : main()
