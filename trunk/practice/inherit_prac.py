class A:
    def __init__(self) -> None:
        print("A")


class B(A):
    def __init__(self) -> None:
        print("B")
        super().__init__()


class C(A):
    def __init__(self) -> None:
        print("C")
        super().__init__()


class D(B, C):
    def __init__(self) -> None:
        print("D")
        super(C, self).__init__()
        super(B, self).__init__()


class E(D):

    def __init__(self) -> None:
        print("E")
        super().__init__()


if __name__ == "__main__":
    inst = E()
