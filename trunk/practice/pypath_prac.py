import os
import sys

__dir_path_str__ = os.path.dirname(__file__)
__dir_name__ = os.path.basename(__dir_path_str__)
WORKSPACE_HOME = __dir_path_str__.replace(f"trunk/{__dir_name__}", "")
sys.path.append(WORKSPACE_HOME)

from core import *
from lc_corr import *
from decompositions import *
from itemcf import *

# 이렇게 쓰는 방법도 있고~~


import argparse


class Validator:
    def __init__(self, *args, **kwargs) -> None:
        for iter in argparse.ArgumentParser(kwargs)._get_kwargs():
            print(iter)
        pass
        self.chain(kwargs)
        self.get_key(kwargs)
        self.get_value(kwargs)
        self.get_test(kwargs)

        """
        print("args")
        for iter in args:
            print(iter)
        print("kwargs")
        self.test(kwargs)
        for k, v in kwargs.items():
            print(f"{k}: {v}")
        """

    # end : public void init()

    def chain(self, kwargs):
        self.get_key(kwargs)
        self.get_value(kwargs)
        self.get_test(kwargs)

    def get_key(self, key: int):
        print("inst.key()")
        print(key)

    def get_value(self, value: int):
        print("inst.value()")
        print(value)

    def get_test(self, target):
        print("inst.target()")
        print(target)


# end : class

if __name__ == "__main__":
    for arg in sys.argv:
        print(f"arg: {arg}")

    '''
    inst = Validator()
    print(type(inst).__name__)
    """
    experiments = Validator(
        key="test",
        value=3,
    )"""

    for inst in DecisionType:
        print(inst)
    '''
# end : main()
