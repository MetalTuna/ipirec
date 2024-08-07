from datetime import datetime

from core import BaseAction


class ColleyAction(BaseAction):

    def __init__(
        self,
        user_id: int = -1,
        item_id: int = -1,
        created_time: str = "",
    ):
        super().__init__(user_id, item_id)
        self.created_time = created_time
        if created_time != "":
            self.timestamp = self.get_timestamp()

    # end : init()

    ## member method
    def get_timestamp(self) -> int:
        """DB의 created_time(DateTime)을 timestamp(UnixTime)로 변환합니다."""
        self.timestamp = (
            0
            if self.created_time == ""
            else int(
                datetime.strptime(
                    self.created_time,
                    "%Y-%d-%m %H:%M:%S",
                ).timestamp(),
            )
        )
        return self.timestamp

    # end : public int get_timestamp()

    @staticmethod
    def record_to_instance(r) -> BaseAction:
        return ColleyAction(
            user_id=int(r["user_id"]),
            item_id=int(r["item_id"]),
            timestamp=int(
                datetime.strptime(
                    str(r["created_time"]),
                    "%Y-%d-%m %H:%M:%S",
                ).timestamp(),
            ),
        )


# end : class
