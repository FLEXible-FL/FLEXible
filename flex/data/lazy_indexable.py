"""
Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import copy
import itertools
import warnings
from collections import deque
from typing import Iterable, Union

import numpy as np


class LazyIndexable:
    def __init__(
        self,
        iterable: Iterable,
        length: int,
        iterable_indexes=None,
        storage: dict = None,
        last_access: deque = None,
    ):
        if iterable_indexes is None:
            iterable_indexes = np.arange(length, dtype=np.uint32)
        if storage is None:
            storage = {}
        self._iterable = iterable
        self._len = length
        self._iterable_indexes = np.asarray(iterable_indexes, dtype=np.uint32)
        self._storage = storage
        if last_access is None:
            self._last_access = deque(maxlen=1)
            self._last_access.append(0)
        else:
            self._last_access = last_access
        try:
            self._iterable[0]
            self._is_generator = False
        except TypeError:  # object is not subscriptable
            self._is_generator = True

    def __repr__(self) -> str:
        return (
            f"len:{self._len}\n"
            f"is_generator:{self._is_generator}\n"
            f"iterable:{self._iterable}\n"
            f"iterable_indexes:{self._iterable_indexes}\n"
            f"storage:{self._storage}"
        )

    def __len__(self) -> int:
        return self._len

    def __getitem_with_seq(self, s: Union[slice, list, np.ndarray]):
        return LazyIndexable(
            self._iterable,
            iterable_indexes=self._iterable_indexes[s],
            length=len(self._iterable_indexes[s]),
            storage=self._storage,
            last_access=self._last_access,
        )

    def __getitem_with_int(self, s):
        index = self._iterable_indexes[s]
        if not self._is_generator:
            return self._iterable[index]
        if index in self._storage:
            return self._storage[index]
        start = self._last_access[-1]
        # If it is not in the storage, we must consume the iterable
        for i, element in enumerate(self._iterable, start=start):
            self._storage[i] = element  # Every we consume is stored for later usage
            self._last_access.append(i + 1)
            if i == index:
                return element
        # Value not found in consumable
        raise IndexError(f"Index {index} out of range")

    def __getitem__(self, s: Union[int, slice, list]):
        #  Proceed with the actual getitem logic
        try:
            s = int(s)
            val = self.__getitem_with_int(s)
        except TypeError:
            val = self.__getitem_with_seq(s)
        return val

    def tolist(self):
        """Function that returns the LazyIndexable as list."""
        return list(self)

    def to_numpy(self, dtype=None):
        if isinstance(self._iterable, (np.ndarray, np.generic)):
            return self._iterable[self._iterable_indexes]
        return np.array(self.tolist(), dtype=dtype)

    def __getstate__(self):
        """Required to make LazyIndexable pickable when self._is_generator==True."""
        state_dict = copy.copy(self.__dict__)
        iterable_is_consumed = False
        for key in state_dict:
            if key == "_iterable" and state_dict["_is_generator"]:
                warnings.warn(  # noqa: B028
                    "Pickling an LazyIndexable fully loads its into memory",
                    RuntimeWarning,
                )
                state_dict[key] = self.tolist()
                iterable_is_consumed = True
        if iterable_is_consumed:  # Ditch is_generator state and _storage
            state_dict["_is_generator"] = False
            state_dict["_storage"] = {}
        return state_dict

    def __deepcopy__(self, memo):
        # """Overwrites deepcopy method."""
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "_iterable" and self._is_generator:
                self._iterable, new_iterable = itertools.tee(self._iterable, 2)
                setattr(result, k, new_iterable)
            else:
                setattr(result, k, copy.deepcopy(v, memo))
        return result
