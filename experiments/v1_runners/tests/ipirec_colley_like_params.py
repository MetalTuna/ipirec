## EMB
import os
import sys
from datetime import datetime

## 3rd Pty.
import json

WORKSPACE_HOME = os.path.dirname(__file__).replace("/experiments", "")
sys.path.append(WORKSPACE_HOME)

## Custom
from core import *
from lc_corr import *

## main.Stack
if __name__ == "__main__":
    dt_now = datetime.now()
    dt_str = dt_now.strftime("%Y%m%d_%H%M%S")
    dataset_dir_path = f"{WORKSPACE_HOME}/data/colley"
    testset_file_path = f"{dataset_dir_path}/like_list.csv"
    results_summary_path = (
        f"{WORKSPACE_HOME}/results/ipa_colley_TopN_like_IRMetrics_{dt_str}.csv"
    )

    # 모델의 매개변수들
    dist_n = 1
    top_n_tags_list = [n for n in range(10, 45, 10)]
    learning_rate_list = [0.1, 0.05, 0.01, 0.005, 0.001]
    generalization_list = [g * 0.1 for g in learning_rate_list]
    co_occur_items_threshold_list = [n for n in range(5, 25, 5)]
    score_iter_list = [n for n in range(20, 60, 20)]
    adjust_iter_list = [n for n in range(20, 50, 10)]
    top_n_conditions = [n for n in range(3, 25, 2)]

    if os.path.exists(results_summary_path):
        os.remove(results_summary_path)

    # 데이터 셋 불러오기
    for top_n_tags in top_n_tags_list:
        for learning_rate in learning_rate_list:
            for generalization in generalization_list:
                for co_occur_items_threshold in co_occur_items_threshold_list:
                    for iterations_threshold in score_iter_list:
                        for adjust_iterations in adjust_iter_list:
                            # 모델 구성하기
                            dataset = ColleyFilteredDataSet(
                                dataset_dir_path=dataset_dir_path
                            )
                            dataset.load_dataset()
                            model_params = CorrelationModel.create_models_parameters(
                                top_n_tags=top_n_tags,
                                co_occur_items_threshold=co_occur_items_threshold,
                                iterations_threshold=iterations_threshold,
                                learning_rate=learning_rate,
                            )
                            jobj_corr_opt = json.dumps(
                                model_params,
                                ensure_ascii=False,
                                indent=4,
                            )
                            model = CorrelationModel(
                                dataset=dataset,
                                model_params=model_params,
                            )
                            model.analysis()

                            # 학습하기
                            model_params = BiasedEstimator.create_models_parameters(
                                learning_rate=learning_rate,
                                generalization=generalization,
                            )
                            jobj_est_opt = json.dumps(
                                model_params,
                                ensure_ascii=False,
                                indent=4,
                            )
                            estimator = BiasedEstimator(
                                model=model,
                                model_params=model_params,
                            )
                            estimator.train(
                                DecisionType.E_VIEW,
                                n=dist_n,
                                emit_iter_condition=adjust_iterations,
                            )
                            estimator.train(
                                DecisionType.E_LIKE,
                                n=dist_n,
                                emit_iter_condition=adjust_iterations,
                            )
                            estimator.train(
                                DecisionType.E_PURCHASE,
                                n=dist_n,
                                emit_iter_condition=adjust_iterations,
                            )

                            # 예측 점수를 기준으로 추천하기
                            recommender = ScoreBasedRecommender(estimator=estimator)
                            recommender.prediction()

                            # 성능평가하기
                            evaluator = IRMetricsEvaluator(
                                recommender=recommender,
                                file_path=testset_file_path,
                            )
                            evaluator.top_n_eval(top_n_conditions=top_n_conditions)
                            evaluator.evlautions_summary_df().to_csv(
                                path_or_buf=results_summary_path,
                                mode="at",
                            )
                            with open(
                                file=results_summary_path,
                                mode="at",
                            ) as fout:
                                fout.write("[Model]\n")
                                fout.write(jobj_corr_opt)
                                fout.write("\n")
                                fout.write("[Estimator]\n")
                                fout.write(jobj_est_opt)
                                fout.write("\n\n")
                                fout.close()
                            # end : StreamWriter(ModelsOpt.)
                        # end : for (score_weight_iters)
                    # end : for (adjust_score_iters)
                # end : for (co_occur_items_conditions)
            # end : for (gamma_options)
        # end : for (lambda_options)
    # end : for (top_n_tags_conditions)


# end : main()
