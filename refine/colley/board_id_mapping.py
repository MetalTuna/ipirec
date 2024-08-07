import pandas as pd
from .base_id_mapping import BaseIDMapping


class BoardIDMapping(BaseIDMapping):
    @property
    def board_id(self) -> int:
        return self._src_id

    def _set_record_(self, r) -> None:
        _id_kwd_str = "item_id"
        self._src_id = int()
        if _id_kwd_str in r:
            self._src_id = int(r[_id_kwd_str])
        else:
            _id_kwd_str = "board_id"
            if _id_kwd_str in r:
                self._src_id = int(r[_id_kwd_str])
            else:
                raise KeyError()

        self.name: str = r["title"]
        self.content: str = r["content"]
        self._tag_string: str = "" if pd.isna(r["tag_string"]) else r["tag_string"]

    # end : protected void set_record()
