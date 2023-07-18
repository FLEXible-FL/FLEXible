from collections import OrderedDict
from typing import Iterable, Union

import numpy as np


class LazyIndexable:
    def __init__(
        self, iterable: Iterable, length: int, iterable_indexes=None, storage=None
    ):
        if iterable_indexes is None:
            iterable_indexes = np.arange(length, dtype=np.uint32)
        if storage is None:
            storage = OrderedDict()
        self._iterable = iterable
        self._len = length
        self._iterable_indexes = np.asarray(iterable_indexes, dtype=np.uint32)
        self._storage = storage
        try:
            self._iterable[0]
            self._is_generator = False
        except TypeError:  # object is not subscriptable
            self._is_generator = True

    def __repr__(self) -> str:
        return (
            f"len:{self._len}\n"
            f"is_generator:{self._is_generator}\n"
            f"iterable_indexes:{self._iterable_indexes}\n"
            f"storage:{self._storage}"
        )

    def __len__(self) -> int:
        return self._len

    def __getitem_with_seq(self, s: Union[slice, list, np.ndarray]):
        return LazyIndexable(
            self._iterable,
            iterable_indexes=self._iterable_indexes[s],
            length=self._len,
            storage=self._storage,
        )

    def __getitem_with_int(self, s):
        index = self._iterable_indexes[s]
        if not self._is_generator:
            return self._iterable[index]
        if index in self._storage:
            return self._storage[index]
        start = next(reversed(self._storage)) + 1 if len(self._storage) > 0 else 0
        # If it is not in the storage, we must consume the iterable
        for i, element in enumerate(self._iterable, start=start):
            self._storage[i] = element  # Every we consume is stored for later usage
            if i == index:
                return element

    def __getitem__(self, s: Union[int, slice, list]):
        #  Proceed with the actual getitem logic
        if not isinstance(s, int):
            return self.__getitem_with_seq(s)
        val = self.__getitem_with_int(s)
        # Value not found in consumable
        if val is None:
            # If the consumable is exhausted, then we now its exact length
            raise IndexError("Index out of range")
        return val

    def to_list(self):
        return list(self)

    def to_numpy(self, dtype=None):
        return np.array(self.to_list(), dtype=dtype)
