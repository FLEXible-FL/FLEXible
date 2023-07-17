import warnings
from collections import OrderedDict
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
                initial_length_guess if length is None else length, dtype=np.uint32
            )
        if storage is None:
            storage = OrderedDict()  # np.empty(len(iterable_indexes), dtype=dtype)
        self._iterable = iterable
        self._len = length
        self._iterable_indexes = np.asarray(iterable_indexes, dtype=np.uint32)
        self._storage = storage
        try:
            self._iterable[0]
            self._is_generator = False
        except TypeError:  # object is not subscriptable
            self._is_generator = True
        if not self._is_generator:
            try:
                self._len = len(self._iterable)
            except TypeError:
                pass

    def __repr__(self) -> str:
        length = "??" if self._len is None else self._len
        return (
            f"len:{length}\n"
            f"len_is_guessed:{length is None}\n"
            f"is_generator:{self._is_generator}\n"
            f"iterable_indexes:{self._iterable_indexes}\n"
            f"storage:{self._storage}"
        )

    def __len__(self, hide_warning=False) -> int:
        if self._len is not None:
            return self._len
        if not hide_warning:
            # trunk-ignore(ruff/B028)
            warnings.warn("The returned value is estimated", RuntimeWarning)
        return len(self._iterable_indexes)

    def _increase_iterable_indexes_len(self, growth_rate=1.6):
        current_max = max(self._iterable_indexes) + 1
        new_indexes = np.arange(
            current_max, round(current_max * growth_rate), dtype=np.uint32
        )
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

    def to_list(self):
        lst = [i for i in self]
        if self._len is None:
            self._len = len(lst)
        return lst

    def to_numpy(self):
        return np.array(self.to_list())
