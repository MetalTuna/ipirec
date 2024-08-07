class MovieEntity:
    def __init__(
        self,
        movie_id: int = -1,
        genre_str: str = "",
        item_id: int = -1,
        tmdb_id: int = -1,
        license_str: str = "",
        title: str = "",
    ):
        self.item_id = item_id
        """기본값: -1"""
        self.movie_id = movie_id
        """기본값: -1"""
        self.tmdb_id = tmdb_id
        """기본값: -1"""
        self.title_str = title
        self.license_str = license_str
        self.license_set = MovieEntity.string_to_set(license_str)
        self._genres = genre_str
        self.genres_set = MovieEntity.string_to_set(genre_str)
        self.request_str: str = None
        self.response_obj = None

    # end : init()

    @property
    def genre_str(self) -> str:
        """
        요약:
            이 영화가 속한 장르들의 값을 가져옵니다. 구분자(|)로 장르들이 분리된 문자열 입니다.
        반환:
            str: 장르 정보가 없다면 공백 문자열이 반환됩니다.
        """
        return self._genres

    @genre_str.setter
    def genre_str(self, genres: str):
        """
        요약:
            장르들의 문자열로 설정하고, 포함된 장르들을 genres_set에 추가합니다.

        매개변수:
            genres (str): 분리자(|)로 구성된 문자열을 입력받습니다.
        """
        self._genres = genres.strip()
        if self._genres == "(no genres listed)" or self._genres == "":
            return
        self.genres_set = MovieEntity.string_to_set(self._genres)

    @staticmethod
    def string_to_set(set_string: str, sep: str = "|") -> set:
        return {e for e in set_string.strip().split(sep) if e != ""}

    # end : public static string string_to_set()

    @staticmethod
    def set_to_string(string_set: set, sep: str = "|") -> str:
        concat_str = ""
        for s in string_set:
            concat_str += s + sep
        # end : for (collections)
        length = len(concat_str)
        if length <= 1:
            return concat_str
        return concat_str[: length - 1]

    # end : public static string set_to_string()


# end : class
