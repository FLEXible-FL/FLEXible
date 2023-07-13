import warnings
from inspect import isgeneratorfunction
from types import GeneratorType
from typing import Iterable, Union

import numpy as np

INIT_LEN_GUESS = 1000


def is_empty_slice(s: slice):
    if isinstance(s, slice):
        if s.start is None and s.stop == 0:
            return True
        if s.start is not None and s.stop is not None and s.start > s.stop:
            return True
    return False


def contains_negative_indexes(s: Union[int, list, np.ndarray]):
    if isinstance(s, (list, np.ndarray)):
        return any(i < 0 for i in s)
    elif isinstance(s, slice):
        return (s.start is not None and s.start < 0) or (
            s.stop is not None and s.stop < 0
        )
    elif isinstance(s, int):
        return s < 0
    raise NotImplementedError(f"Indexing with type {type(s)} is not supported")


class LazyIndexable:
    def __init__(
        self,
        iterable: Iterable,
        length=None,
        iterable_indexes=None,
        storage=None,
        initial_length_guess=INIT_LEN_GUESS,
    ):
        if iterable_indexes is None:
            iterable_indexes = np.arange(
                initial_length_guess if length is None else length,
                dtype=np.uint32
            )
        if storage is None:
            storage = {}
        self._iterable = iterable
        self._len = length
        self._iterable_indexes = np.asarray(iterable_indexes, dtype=np.uint32)
        self._storage = storage
        # This last check is a little hacky
        self._is_generator = (
            isinstance(self._iterable, GeneratorType)
            or isgeneratorfunction(self._iterable)
            or "iterator" in type(self._iterable).__name__
        )

    def __repr__(self) -> str:
        return (
            f"len:{len(self)}\n"
            f"len_is_guessed:{self._len is None}\n"
            f"iterable_indexes:{self._iterable_indexes}\n"
            f"storage:{self._storage}"
        )

    def __len__(self, hide_warning=False) -> int:
        if self._len is None and not hide_warning:
            # trunk-ignore(ruff/B028)
            warnings.warn("The returned value is estimated", RuntimeWarning)
        return len(self._iterable_indexes)

    def _increase_iterable_indexes_len(self, growth_rate=1.6):
        current_max = max(self._iterable_indexes) + 1
        new_indexes = np.arange(current_max, round(current_max * growth_rate), dtype=np.uint32)
        self._iterable_indexes = np.concatenate(
            (self._iterable_indexes, new_indexes), axis=0
        )

    def __getitem_with_seq(self, s: Union[slice, list, np.ndarray]):
        return LazyIndexable(
            self._iterable,
            iterable_indexes=self._iterable_indexes[s],
            length=self._len,
            storage=self._storage,
        )

    def __getitem_with_int(self, s):
        index = self._iterable_indexes[s]
        if index in self._storage:
            return self._storage[index]
        start = (
            max(self._storage) + 1
            if self._is_generator and len(self._storage) > 0
            else 0
        )
        # If it is not in the storage, we must consume the iterable
        for i, element in enumerate(self._iterable, start=start):
            self._storage[i] = element  # Every we consume is stored for later usage
            if i in self._iterable_indexes and i == index:
                return element

    def __getitem__(self, s: Union[int, slice, list]):
        #  Safety checks
        if self._len is None:
            if contains_negative_indexes(s):
                raise IndexError(
                    "Negative indexing is not supported if length is not provided."
                )
            if is_empty_slice(s):
                return []
        #  Ensure there is enough space in self._iterable_indexes
        loop_count = 0
        continue_loop = True
        while continue_loop and self._len is None:
            try:
                if isinstance(s, int):
                    self._iterable_indexes[s]
                elif len(self._iterable_indexes[s]) == 0:
                    raise IndexError()
                continue_loop = False
            except IndexError:
                self._increase_iterable_indexes_len()
            loop_count += 1
            if loop_count % 20 == 0:
                # trunk-ignore(ruff/B028)
                warnings.warn(
                    "It might be stuck in an infinite loop, try providing the length or increasing the initial_len_guess",
                    RuntimeWarning,
                )

        #  Proceed with the actual getitem logic
        if not isinstance(s, int):
            return self.__getitem_with_seq(s)
        val = self.__getitem_with_int(s)
        # Value not found in consumable
        if val is None:
            # If the consumable is exhausted, then we now its exact length
            if self._len is None:
                self._iterable_indexes = self._iterable_indexes[: max(self._storage)]
                self._len = len(self._iterable_indexes)
            raise IndexError("Index out of range")

        return val

    # Better use list()
    # def to_list(self):
    #     lst = list(self)
    #     if self._len is None:
    #         self._len = len(lst)
    #     return lst

    # better use np.array()
    # def to_numpy(self):
    #     lst = np.array(self)
    #     if self._len is None:
    #         self._len = len(lst)
    #     return lst
