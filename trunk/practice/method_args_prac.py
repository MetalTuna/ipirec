class ArgumentsPractice:

    def __init__(self, *args, **kwargs) -> None:
        # self.enumarable_args_method(args)
        self.keyword_args_method(kwargs)

    def enumarable_args_method(self, key, value, *args):
        print(key)
        print(value)
        print(args)

    def keyword_args_method(self, **kwargs):
        print(kwargs["key"])
        print(kwargs["value"])
        print(kwargs)


if __name__ == "__main__":
    values_collection = [1, "asdasdd", "value", 577.25]
    values_dict = {
        "key": "value_pair",
        "value": 30,
        "learning_rate": 0.0005,
    }

    inst = ArgumentsPractice(values_dict)
