import os
import pandas as pd

from core import ItemEntity


class MovieLensItemEntity(ItemEntity):

    def __init__(
        self,
        item_id: int = -1,
        movie_id: int = -1,
        tags_set: set = set(),
    ):
        self.movie_id = movie_id
        super().__init__(item_id, tags_set)

    @staticmethod
    def load_collection(dataset_dir_path: str) -> dict:
        """_summary_

        Args:
            dataset_dir_path (str): ${DATA_HOME}

        Returns:
            dict: _description_
        """

        # dir_path = /Users/taegyu.hwang/Documents/tghwang_git_repo/ipitems_analysis/data/ml
        # files = [view, like, purchase]_list.csv, item_list.csv
        file_path = f"{dataset_dir_path}/item_list.csv"
        if not os.path.exists(file_path):
            raise FileNotFoundError()
        item_dict = dict()
        movie_id_to_item_id_dict = dict()

        for iter in pd.read_csv(file_path).itertuples(index=False):
            inst = MovieLensItemEntity.parse_entity(iter)
            movie_id_to_item_id_dict.update({inst.movie_id: inst.item_id})
            item_dict.update({inst.item_id: inst})
        # end : for (items)

        file_path = f"{dataset_dir_path}/movies.csv"
        for _, r in pd.read_csv(file_path).iterrows():
            movie_id = int(r["movieId"])
            if not movie_id in movie_id_to_item_id_dict:
                continue
            item_id: int = movie_id_to_item_id_dict[movie_id]
            inst: MovieLensItemEntity = item_dict[item_id]
            inst.item_name = str(r["title"])
        # end : for (movies)
        return item_dict

    @staticmethod
    def parse_entity(iter):
        return MovieLensItemEntity(
            item_id=int(iter.item_id),
            movie_id=int(iter.movie_id),
            tags_set={l for l in iter.license.split("|")}.union(
                {g for g in iter.genre.split("|")}
            ),
        )
