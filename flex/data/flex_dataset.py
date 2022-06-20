from collections import UserDict
from collections.abc import Sized
from typing import Any, List, Optional

import numpy as np
import numpy.typing as npt


def same_length_check(obj1: Sized, obj2: Sized):
    if len(obj1) != len(obj2):
        raise ValueError(
            f"Provided arguments must have the same length, but length={len(obj1)} and length={len(obj2)} were given."
        )


class FlexDataObject:
    """Class used to represent the dataset from a client in a Federated Learning enviroment.

    Attributes
    ----------
    X_data: numpy.typing.ArrayLike
        A numpy.array containing the data for the client.
    y_data: numpy.typing.ArrayLike
        A numpy.array containing the labels for the training data. Can be None if working
        on an unsupervised learning task. Default None.
    X_names: Optional[List[str]]
        A list of strings containing the names of the features set. Default None.
    y_names: Optional[List[str]]
        A list of strings containing the class names. Default None.
    """

    __slots__ = ("__X_data", "__y_data", "__X_names", "__y_names")

    def __init__(
        self,
        X_data: npt.NDArray,
        y_data: Optional[npt.NDArray] = None,
        X_names: Optional[List[str]] = None,
        y_names: Optional[List[str]] = None,
    ) -> None:

        self.__X_data = X_data
        self.__y_data = y_data
        self.__X_names = X_names
        self.__y_names = y_names

        if X_names is not None:
            same_length_check(X_data[0], X_names)

        if y_names is not None and y_data is not None:
            same_length_check(np.unique(y_data, axis=0), y_names)

        if y_names is not None:
            same_length_check(X_data, y_data)

    @property
    def X_data(self):
        return self.__X_data

    @property
    def X_names(self):
        return self.__X_names

    def setX(self, X_data: npt.NDArray, X_names: Optional[List[str]] = None) -> None:
        if X_names is not None:
            same_length_check(X_data[0], X_names)
        self.__X_data = X_data
        self.__X_names = X_names

    @property
    def y_data(self):
        return self.__y_data

    @property
    def y_names(self):
        return self.__y_names

    def setY(self, y_data: npt.NDArray, y_names: Optional[List[str]] = None) -> None:
        if y_names is not None:
            same_length_check(np.unique(y_data, axis=0), y_names)
        self.__y_data = y_data
        self.__y_names = y_names


class FlexDataset(UserDict):
    """Class that represents a federated dataset for the Flex library.
    The dataset contains the ids of the clients and the dataset associated
    with each client.

    Attributes
    ----------
    data (collections.UserDict): The structure is a dictionary
        with the clients ids as keys and the dataset as value.
    """

    def __setitem__(self, key: str, item: FlexDataObject) -> None:
        self.data[key] = item

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default
