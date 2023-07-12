from collections import OrderedDict
from inspect import isgeneratorfunction
from types import GeneratorType
from typing import Iterable, Union

import numpy as np


class LazyIndexable:
    def __init__(self, iterable: Iterable, length, iterable_indexes=None, storage=None):
        if iterable_indexes is None:
            iterable_indexes = np.arange(length)
        if storage is None:
            storage = OrderedDict()
        self._iterable = iterable
        self._len = length
        self._iterable_indexes = np.asarray(iterable_indexes)
        self._storage = storage
        # This last check is a little hacky
        self._is_generator = (
            isinstance(self._iterable, GeneratorType)
            or isgeneratorfunction(self._iterable)
            or "iterator" in type(self._iterable).__name__
        )

    def __repr__(self) -> str:
        return (
            f"len:{self._len}\n"
            f"iterable_indexes:{self._iterable_indexes}\n"
            f"storage:{self._storage}"
        )

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, s: Union[int, slice, list]):
        if not isinstance(s, int):
            return LazyIndexable(
                self._iterable,
                iterable_indexes=self._iterable_indexes[s],
                length=len(self._iterable_indexes[s]),
                storage=self._storage,
            )

        index = self._iterable_indexes[s]
        if index in self._storage:
            return self._storage[index]
        start = (
            max(self._storage.keys()) + 1
            if self._is_generator and len(self._storage) > 0
            else 0
        )
        # If it is not in the storage, we must consume the iterable
        for i, element in enumerate(self._iterable, start=start):
            self._storage[i] = element  # Every we consume is stored for later usage
            if i in self._iterable_indexes and i == index:
                return element
        raise IndexError("Index out of range")

    def to_numpy(self):
        # Consume the entire iterable if possible
        start = (
            max(self._storage.keys()) + 1
            if self._is_generator and len(self._storage) > 0
            else 0
        )
        for i, element in enumerate(self._iterable, start=start):
            if i not in self._storage:
                self._storage[i] = element
        tmp_list = [self._storage[i] for i in self._iterable_indexes]
        return np.asarray(tmp_list)
