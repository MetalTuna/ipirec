# Std. LIB.
import os
from multiprocessing import Pool

# 3rd Pty. LIB.
from tqdm import tqdm
import numpy as np

# Cus. LIB.
from core import BaseRecommender, BaseTrain, BaseAction, UserEntity


class ParallelPredictRecommender(BaseRecommender):
    """
    사용하지 않음
    =====
    사유: 실행결과가 다름
    """

    def __init__(
        self,
        estimator: BaseTrain,
    ) -> None:
        self.__predicted_scores_array: np.ndarray = None
        """예측 값으로 채워질 행렬"""
        super().__init__(estimator)
        self._vcore: int = os.cpu_count() - 1
        """사용할 수 있는 코어 수"""

    def _preprocess_(self) -> None:
        super()._preprocess_()
        self.__predictions_array_alloc__()

    # end : protected override void preprocess()

    def __predictions_array_alloc__(self) -> None:
        self.__predicted_scores_array = np.zeros(
            shape=(self.users_count, self.items_count),
            dtype=np.float16,
        )

    # end : private void p_array_alloc()

    def prediction(self) -> None:
        users_len = len(self.user_id_to_idx.keys())
        # predict_items_len = len(self.item_id_to_idx)
        tqdm_desc_str = "ParallelPredict()"
        if self._is_predicted:
            return
        # for user_id in self.user_id_to_idx.keys():
        items_list: list = self.item_id_to_idx.keys()
        for user_id in tqdm(
            iterable=self.user_id_to_idx.keys(),
            desc=tqdm_desc_str,
            total=users_len,
        ):
            ## [Prediction] ParallelFor.ForAllItems()
            # user 단위로 분기하는 것이 병목이 적을 것 같음
            predict_list: list = [
                BaseAction(user_id, item_id) for item_id in items_list
            ]
            with Pool() as p:
                predict_list = p.map(
                    self._estimator._estimate_,
                    iterable=predict_list,
                )
                p.join()
            # end : Pool()
            """
            args_list: list = [(user_id, item_id) for item_id in items_list]
            with Pool() as p:
                p.starmap(
                    self.predict,
                    iterable=args_list,
                )
                p.join()
                p.close()
            # end : Pool()
            """

            """
            process_list = list()
            for item_id in self.item_id_to_idx.keys():
                p = Process(target=self.predict, args=(user_id, item_id))
                p.start()
                process_list.append(p)
            # end : for (items) -- ParallelFor
            for p in process_list:
                p: Process
                p.join()
            # end : for (tasks) -- wait()
            process_list.clear()
            """
            ## [Casting] np.array => predicted_items : list(BaseAction)
            user: UserEntity = self.user_dict[user_id]
            user.estimated_items_score_list = predict_list
        # end : for (users)
        self._is_predicted = True

    # end : public void prediction()

    def predict(
        self,
        user_id: int,
        item_id: int,
    ) -> float:
        # 있는 것만 분기하도록 구현했지만.... 만약을 위해 예외처리
        uidx: int = self.user_id_to_idx.get(user_id, -1)
        iidx: int = self.item_id_to_idx.get(item_id, -1)
        if uidx == -1 or iidx == -1:
            return 0.0
        self.__predicted_scores_array[uidx, iidx] = self._estimator.predict(
            user_id, item_id
        )
        return self.__predicted_scores_array[uidx, iidx]

    # end : public override float predict()


# end : class
