## 3rd Pty. LIB.
import pandas as pd

## Custom LIB.
from .product_id_mapping import ProductIDMapping
from .colley_repository import ColleyRepository


class ColleyRepositoryAppendScrap(ColleyRepository):

    def __convert__(self, export_dir_path: str = "") -> None:
        export_dir_path = (
            self._raw_data_path if export_dir_path == "" else export_dir_path
        )
        super().__convert__(export_dir_path)
        self.product_item_dict

        ## append product_scrap_list
        with open(
            file=f"{export_dir_path}/purchase_list.csv",
            mode="at+",
            encoding="utf-8",
        ) as fout:
            for _, r in pd.read_csv(
                filepath_or_buffer=f"{self._raw_data_path}/product_scrap_list.csv",
            ).iterrows():
                product_id = int(r["product_id"])
                if not product_id in self.product_item_dict:
                    continue
                inst: ProductIDMapping = self.product_item_dict[product_id]
                user_id = int(r["user_id"])
                created_time = str(r["created_time"]).strip()
                item_id = inst.item_id
                fout.write(
                    str.format(
                        "{0}, {1}, {2}\n",
                        user_id,
                        item_id,
                        created_time,
                    )
                )
            fout.close()
        # end : StreamWriter()

    # end : private void convert()

    def __clone__(self) -> None:
        super().__clone__()
        # 스크랩
        self.get_raw_data(self.SQL_PRODUCT_SCRAP_LIST).to_csv(
            f"{self._raw_data_path}/product_scrap_list.csv"
        )

    # end : private void clone()


# end : class
