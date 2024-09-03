import time
import requests

from tqdm import tqdm
import pandas as pd
import json

from core import BaseRepository, Machine
from .movie_entity import MovieEntity

### 남은 일
# [재실행 관련처리]
# 라이센스 없는 것 추가하기
# 영화장르 없는 것 추가하기


class MovieLensRepository(BaseRepository):
    """
    - 요약:
        - MovieLens Latest Datasets의 small을 사용합니다.
            - Doc: https://files.grouplens.org/datasets/movielens/ml-latest-small-README.html
            - Src: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
        - 원시 데이터 가공
            - 영화: 메타데이터를 추가합니다.
                - 영화제작사 (토이스토리->픽사)
                - 영화장르 (빵꾸난 것 있으면 채우는 것)
            - 평점: 점수를 3개의 의사결정으로 변환합니다.
                - 봤다: 1, 2, 3
                - 좋다: 3, 4
                - 샀다: 4, 5
    - 원시데이터 구성
        - 공통
            - 규격: 쉼표로 구분됨
        - 파일
            - 사용자: 없음
            - 영화(movies.csv): movieid,title,genres
                - movieid(int): 1
                - title(str): Toy Story (1995)
                - genres(str): Adventure|Animation|Children|Comedy|Fantasy
            - 영화의 참조정보(links.csv): movieId,imdbId,tmdbId
                - ex. 1,114709,862
            - 태그: userId,movieId,tag,timestamp
                - ex. 2,60756,funny,1445714994
            - 평점: userId,movieId,rating,timestamp
                - rating: 0.5:5.0:0.5
    """

    def __init__(
        self,
        raw_data_path: str,
        db_src: Machine = Machine.E_MAC,
        tmdb_auth_key_str: str = "YOUR_TMDB_API_PKEY",
    ) -> None:
        super().__init__(
            raw_data_path,
            db_src,
        )

        self.success_dict: dict = None
        """
        Key: movie_id (int)
        Value: instance (MovieEntity)
        """
        self.failure_dict: dict = None
        """
        Key: movie_id (int)
        Value: instance (MovieEntity)
        """

        '''
        # self.item_dict = dict()
        """
        Key: item_id (int)
        Value: instance (MovieEntity)
        """
        '''

        self.__headers = {
            "accept": "application/json",
            "Authorization": tmdb_auth_key_str,
        }
        """request_session_key (str)"""
        self.__list_of_decision_rules = [[0.5, 3.0], [3.0, 4.0], [4.0, 5.0]]
        """
        - Default OPT.
            - [0] view: 0.5 -- 3.0
            - [1] like: 3.0 -- 4.0
            - [2] purchase: 4.0 -- 5.0
        """

    def convert_decision(
        self,
        raw_rating_file_path: str = "",
        export_dir_path: str = "",
    ):
        """
        요약:
            원시 평점데이터를 항목구성에 맞게 정제합니다.

        매개변수:
            raw_rating_file_path (str, optional): 원시평점 파일주소(절대경로). Defaults to "".
            export_dir_path (str, optional): 정제된 의사결정이 저장될 폴더주소(절대경로). Defaults to "".
        """
        raw_rating_file_path = (
            raw_rating_file_path
            if raw_rating_file_path != ""
            else f"{self._raw_data_path}/ratings.csv"
        )
        export_dir_path = (
            export_dir_path if export_dir_path != "" else self._raw_data_path
        )

        items_df = pd.read_csv(
            filepath_or_buffer=f"{self._raw_data_path}/item_list.csv",
            sep=",",
        )
        movie_item_dict = dict()
        for _, r in items_df.iterrows():
            item_id = int(r["item_id"])
            movie_id = int(r["movie_id"])
            license = str(r["license"])
            genre = str(r["genre"])
            if item_id == -1:
                raise NotImplementedError()
                continue
            inst = MovieEntity(
                movie_id=movie_id,
                genre_str=genre,
                item_id=item_id,
                license_str=license,
            )
            # inst = MovieEntity(item_id, movie_id, license, genre)
            movie_item_dict.update({movie_id: inst})
        # end : for (items)

        ratings_df = pd.read_csv(filepath_or_buffer=raw_rating_file_path)
        # userId,movieId,rating,timestamp
        with open(
            file=f"{export_dir_path}/view_list.csv", mode="wt", encoding="utf-8"
        ) as vf, open(
            file=f"{export_dir_path}/like_list.csv", mode="wt", encoding="utf-8"
        ) as lf, open(
            file=f"{export_dir_path}/purchase_list.csv", mode="wt", encoding="utf-8"
        ) as pf:
            line = "user_id,item_id,rating,timestamp\n"
            vf.write(line)
            lf.write(line)
            pf.write(line)
            for _, r in ratings_df.iterrows():
                movie_id = int(r["movieId"])
                if not movie_id in movie_item_dict:
                    continue
                inst: MovieEntity = movie_item_dict[movie_id]
                user_id = int(r["userId"])
                item_id = inst.item_id
                rating = float(r["rating"])
                timestamp = int(r["timestamp"])
                line = str.format(
                    "{0},{1},{2},{3}\n", user_id, item_id, rating, timestamp
                )

                # 봤다는 의사결정의 긍정과 무관하게 기록해야함
                vf.write(line)
                if (
                    rating >= self.__list_of_decision_rules[0][0]
                    and rating <= self.__list_of_decision_rules[0][1]
                ):
                    continue
                elif (
                    rating >= self.__list_of_decision_rules[1][0]
                    and rating <= self.__list_of_decision_rules[1][1]
                ):
                    lf.write(line)
                else:
                    pf.write(line)
                """
                # [24.05.02] 아... 의사결정 별로 구분해서 출력함...
                if (
                    rating >= self.__list_of_decision_rules[0][0]
                    and rating <= self.__list_of_decision_rules[0][1]
                ):
                    vf.write(line)
                elif (
                    rating >= self.__list_of_decision_rules[1][0]
                    and rating <= self.__list_of_decision_rules[1][1]
                ):
                    lf.write(line)
                else:
                    pf.write(line)
                """
            # end : for (ratings)
            pf.close()
            lf.close()
            vf.close()
        # end : StreamWriter(views, likes, purchases)

    # end : public override void convert_decision()

    def dump_data(self):
        """
        - 요약:
            - 영화 메타데이터 복제
                - 라이센스, 장르 값이 모두 있는 영화들로 항목을 구성합니다.
                - 메타데이터 수집에 실패한 영화들을 기록합니다.
        """
        # 항목내역
        file_path = f"{self._raw_data_path}/item_list.csv"
        with open(file=file_path, mode="wt", encoding="utf-8") as fout:
            line = "item_id,movie_id,license,genre\n"
            item_id = 1
            fout.write(line)
            for movie_id in self.success_dict.keys():
                inst: MovieEntity = self.success_dict[movie_id]
                line = str.format(
                    '{0},{1},"{2}","{3}"\n',
                    item_id,
                    inst.movie_id,
                    inst.license_str,
                    inst.genre_str,
                )
                fout.write(line)
                item_id += 1
            # end : for (success_items)
            fout.close()
        # end : StreamWriter()

        # 실패내역
        file_path = f"{self._raw_data_path}/failure_list.csv"
        with open(file=file_path, mode="wt", encoding="utf-8") as fout:
            line = ",movie_id, tmdb_id, response\n"
            item_id = 1
            fout.write(line)
            for movie_id in self.failure_dict.keys():
                inst: MovieEntity = self.failure_dict[movie_id]
                line = str.format(
                    '{0},{1},{2},"{3}"\n',
                    item_id,
                    inst.movie_id,
                    inst.tmdb_id,
                    inst.response_obj,
                )
                fout.write(line)
                item_id += 1
            # end : for (failure_items)
            fout.close()
        # end : StreamWriter()

    # end : dump_data()

    def load_data(self):
        """
        - 원시데이터 구성
        - 공통
            - 규격: 쉼표로 구분됨
        - 파일
            - 사용자: 없음
            - 영화(movies.csv): movieid,title,genres
                - movieid(int): 1
                - title(str): Toy Story (1995)
                - genres(str): Adventure|Animation|Children|Comedy|Fantasy
            - 영화의 참조정보(links.csv): movieId,imdbId,tmdbId
                - ex. 1,114709,862
            - 태그: userId,movieId,tag,timestamp
                - ex. 2,60756,funny,1445714994
            - 평점: userId,movieId,rating,timestamp
                - rating: 0.5:5.0:0.5
        """
        movie_df = pd.read_csv(f"{self._raw_data_path}/movies.csv")
        links_df = pd.read_csv(f"{self._raw_data_path}/links.csv")
        movie_links_df = movie_df.merge(links_df, how="inner", on="movieId")

        success_dict = dict()
        failure_dict = dict()

        # MaxReq/PS = 50
        lps_delay = 1 / 50

        ### 실행조건
        # 최초 실행인가?
        # for _, r in movie_links_df.iterrows():
        for _, r in tqdm(
            iterable=movie_links_df.iterrows(),
            desc="get movies metadata",
            total=movie_links_df.shape[0],
        ):
            movie_id = int(r["movieId"])
            tmdb_id = -1 if pd.isna(r["tmdbId"]) else int(r["tmdbId"])
            genres = "" if pd.isna(r["genres"]) else str(r["genres"])
            inst = MovieEntity(
                movie_id=movie_id,
                genre_str=genres,
                tmdb_id=tmdb_id,
            )
            # inst = MovieEntity(movie_id, tmdb_id, genres)
            if tmdb_id == -1:
                failure_dict.update({movie_id: inst})
                continue
            request_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}?language=en-US"
            try:
                response = requests.get(request_url, headers=self.__headers)
                time.sleep(lps_delay)
            except Exception as e:
                inst.response_obj = e
                failure_dict.update({movie_id: inst})
                continue
            inst.response_obj = response
            # 이걸 안하니 요청을 거절함

            if response.status_code != 200:
                inst.response_obj = response
                failure_dict.update({movie_id: inst})
                continue

            jobj = json.loads(response.text)
            # append licenses
            for obj in jobj["production_companies"]:
                ip_str = obj["name"].strip()
                inst.license_set.add(ip_str)
            inst.license_str = MovieEntity.set_to_string(inst.license_set)

            if len(inst.genres_set) == 0:
                if "genres" in jobj:
                    for g in {obj["name"].strip() for obj in jobj["genres"]}:
                        inst.genres_set.add(g)
                    # end : for (genres)
                    inst._genres = MovieEntity.set_to_string(inst.genres_set)

            if len(inst.license_set) != 0 and len(inst.genres_set) != 0:
                success_dict.update({movie_id: inst})
            else:
                failure_dict.update({movie_id: inst})
        # end : for (movies)

        self.success_dict = success_dict
        self.failure_dict = failure_dict

    # end : public void load_data()


# end : class
