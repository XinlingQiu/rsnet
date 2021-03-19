from typing import Iterable
from itertools import repeat
import os
import os.path as osp


def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


pair = _ntuple(2)


def mkdir(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)
