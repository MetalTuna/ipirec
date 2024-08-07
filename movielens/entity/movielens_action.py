from core import BaseAction


class MovieLensAction(BaseAction):

    def __init__(
        self,
        user_id: int = -1,
        item_id: int = -1,
        timestamp: int = 0,
    ):
        super().__init__(user_id, item_id)
        self.timestamp = timestamp

    @staticmethod
    def record_to_instance(r) -> BaseAction:
        return MovieLensAction(
            user_id=int(r["user_id"]),
            item_id=int(r["item_id"]),
            # timestamp=int(r["timestamp"]),
            timestamp=int(r["created_time"]),
        )
