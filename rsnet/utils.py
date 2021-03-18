from typing import Iterable
from itertools import repeat


def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


pair = _ntuple(2)
