from threading import Thread
from tqdm import tqdm

from core import BaseDataSet
from .base_distance_model import BaseDistanceModel


class ParallelDistanceModel(BaseDistanceModel):
    """
    - 사용중지
    =====
        - 사유: 실행결과가 다르므로, 결과를 보장할 수 없다.

    - 요약:
        - 유사도 계산의 병렬화를 위한 클래스입니다.
    """

    def __init__(self, dataset: BaseDataSet) -> None:
        super().__init__(dataset)

    # end : init()

    def _process_(self) -> None:
        """
        - 요약:
            - Pool을 사용해서, 유사도 계산을 다중분기합니다.
        """

        # 유사도 계산
        item_ids_list = list(self.item_id_to_idx.keys())
        items_len = len(item_ids_list)

        # for item_x in self.item_id_to_idx.keys():
        for item_x in tqdm(
            iterable=self.item_id_to_idx.keys(),
            desc="ParallelDistanceModel.process()",
            total=items_len,
        ):
            idx = self.item_id_to_idx[item_x]
            self.arr_similarties[idx][idx] = 1.0
            item_ids_list.remove(item_x)
            # end : for (I - x)

            """
            process_list = [
                Process(group=None, target=self.__distance__, args=(item_x, item_y))
                for item_y in item_ids_list
            ]
            """
            process_list = list()
            for item_y in item_ids_list:
                p = Thread(
                    target=self.__distance__,
                    args=(
                        item_x,
                        item_y,
                    ),
                )
                p.start()
                process_list.append(p)
            # end : for (items)
            for p in process_list:
                p: Thread
                p.join()
            # end : for (items)

            """
            iterable_args = [(item_x, item_y) for item_y in item_ids_list]
            with Pool() as p:
                p.starmap(
                    func=self.__distance__,
                    iterable=iterable_args,
                )
                # p.join()
            # end : Pool()
            """
        # end : for (items)

    # end : protected override void process()


# end : class
