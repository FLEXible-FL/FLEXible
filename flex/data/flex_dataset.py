from collections import UserDict
from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import numpy.typing as npt


@dataclass
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

    X_data: npt.NDArray
    y_data: Optional[npt.NDArray] = None
    X_names: Optional[List[str]] = None
    y_names: Optional[List[str]] = None

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, pos):
        if self.y_data is not None:
            return (self.X_data[pos], self.y_data[pos])
        else:
            return (self.X_data[pos], None)

    def validate(self):
        """Function that checks whether the object is correct or not."""
        if self.y_data is not None and len(self.X_data) != len(self.y_data):
            raise ValueError(
                f"X_data and y_data must have equal lenght. X_data has {len(self.X_data)} elements and y_data has {len(self.y_data)} elements."
            )
        if self.X_names is not None and len(self.X_data[0]) != len(self.X_names):
            raise ValueError(
                f"X_data and X_names has different lenght and they must have the same. X_data has len {len(self.X_data[0])} and X_names has len {len(self.X_names)}."
            )
        if (
            self.y_names is not None
            and self.y_data is not None
            and len(np.unique(self.y_data, axis=0)) != len(self.y_names)
        ):
            raise ValueError(
                f"y_data has differents unique values that y_names values. y_data has {len(np.unique(self.y_data, axis=0))} unique values, and y_names has {len(self.y_names)}."
            )
        if self.y_data is not None and self.y_data.ndim > 1:
            raise ValueError(
                "y_data is multidimensional and we only support unidimensional labels."
            )


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
