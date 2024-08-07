class ItemsMapping:
    def __init__(self, item_id: int, name: str):
        self.item_id = item_id
        self.name = name
        self.board_ids_set = set()
        self.product_ids_set = set()
        self.tags_set = set()
