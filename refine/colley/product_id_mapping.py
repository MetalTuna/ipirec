import pandas as pd
from .base_id_mapping import BaseIDMapping


class ProductIDMapping(BaseIDMapping):
    @property
    def product_id(self) -> int:
        return self._src_id

    def _set_record_(self, r) -> None:
        self._src_id = r["product_id"]
        self.name: str = r["product_name"]
        self._tag_string: str = "" if pd.isna(r["tag_string"]) else r["tag_string"]
