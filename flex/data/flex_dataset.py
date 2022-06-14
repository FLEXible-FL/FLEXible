from collections import UserDict
from typing import Optional, List, Sequence, Any

import numpy as np
import numpy.typing as npt


class FlexClientDataset:
    """Class used to represent the dataset from a client in a Federated Learning enviroment.
    
    Attributes
    ----------
    X_data: numpy.typing.NDArray
        A numpy.array containing the data for the client.
    y_data: Optional[Sequence[numpy.typing._SupportsArray[numpy.dtype[Any]]]]
        A numpy.array containing the labels for the training data. Can be None if working
        on unsupervised learning problem. Default None.
    X_names: Optional[List[str]]
        A list of strings containing the names of the features set. Default None.
    y_names: Optional[List[str]]
        A list of strings containing the class names. Default None.
    """
    __slots__ = ('__X_data', '__y_data', '__X_names', '__y_names')
    
    def __init__(self, X_data: npt.NDArray, y_data: Optional[Sequence[npt._SupportsArray[np.dtype[Any]]]] = None, 
                 X_names: Optional[List[str]] = None, y_names: Optional[List[str]] = None) -> None:

        self.__X_data = X_data
        self.__y_data = y_data

        self.__X_names = X_names if (X_names and len(X_names) == X_data.shape[1]) else None
        self.__y_names = y_names if y_names and y_data is not None and (len(np.unique(y_data)) == len(np.unique(y_names))) else None

    @property
    def X_data(self):
        return self.__X_data

    @X_data.setter
    def X_data(self, X_data):
        self.__X_data = X_data

    @property
    def y_data(self):
        return self.__y_data

    @y_data.setter
    def y_data(self, y_data):
        self.__y_data = y_data

    @property
    def X_names(self):
        return self.__X_names

    @X_names.setter
    def X_names(self, X_names):
        self.__X_names = X_names

    @property
    def y_names(self):
        return self.__y_names

    @y_names.setter
    def y_names(self, y_names):
        self.__y_names = y_names


class FlexDataset(UserDict):
    """Class that represent a federated dataset for the Flex library.
    The dataset contains the ids of the clients and the dataset associated
    with each client.

    Attributes
    ----------
    data (collections.UserDict): The structure is a dictionary
        with the clients ids as keys and the dataset as value.
    """

    def __setitem__(self, key: str, item: FlexClientDataset) -> None:
        self.data[key] = item

    def get(self, key:str, default:Optional[Any]=None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default
