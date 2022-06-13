from collections import UserDict
from typing import _KT, _VT, Optional, List, Sequence, Any

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
    client_id: str
        The client's id. Default client_id.
    """
    __slots__ = ('__X_data', '__y_data', '__X_names', '__y_names', '__client_id')
    
    def __init__(self, X_data: npt.NDArray, y_data: Optional[Sequence[npt._SupportsArray[np.dtype[Any]]]] = None, 
                 X_names: Optional[List[str]] = None, y_names: Optional[List[str]] = None, 
                 client_id: Optional[str] = 'client_id') -> None:

        self.__X_data = X_data
        self.__y_data = y_data

        self.__X_names = X_names if (X_names and len(X_names) == X_data.shape[1]) else [f"x{i}" for i in range(X_data.shape[1])]

        if y_names and y_data and (np.unique(y_data) == np.unique(y_names)):
            self.__y_names = y_names
        elif y_data and not y_names:
            self.__y_names = [f"class_{c}" for c in np.unique(y_data)]
        else:
            self.__y_names = ["Not available"] # Must keep the typing

        self.__client_id = client_id

    @property
    def X_data(self):
        return self.__X_data

    @property
    def y_data(self):
        return self.__y_data

    @property
    def X_names(self):
        return self.__X_names

    @property
    def y_names(self):
        return self.__y_names

    @property
    def client_id(self):
        return self.__client_id
    
    
class FlexDataset(UserDict):
    """Class that represent a federated dataset for the Flex library.
    The dataset contains the ids of the clients and the dataset associated
    with each client.

    Attributes
    ----------
    data (collections.UserDict): The structure is a dictionary
        with the clients ids as keys and the dataset as value.
    """
    def __missing__(self, key):
        if not isinstance(key, 'str'):
            raise KeyError
        return self[key]

    def __setitem__(self, key: _KT, item: _VT) -> None:
        self.data[key] = item

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __contains__(self, key: object) -> bool:
        return key in self.data