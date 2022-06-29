from collections import UserDict
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool
from typing import Any, Callable, Hashable, List, Optional

import numpy as np
import numpy.typing as npt

from flex.data.flex_preprocessing_utils import normalize, one_hot_encoding


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

    def __setitem__(self, key: Hashable, item: FlexDataObject) -> None:
        self.data[key] = item

    def get(self, key: Hashable, default: Optional[Any] = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def map(
        self,
        clients_ids: List[Hashable] = None,
        num_proc: int = None,
        func: Callable = None,
        *args,
        **kwargs,
    ):
        """This function applies a custom function to the FlexDataset in parallel.

        The *args and the **kwargs provided to this function are all the args and kwargs
        of the custom function provided by the client.

        Args:
            fld (FlexDataset): FlexDataset containing all the data from the clients.
            clients_ids (List[Hashtable], optional): List containig the the clients id where func will
            be applied. Each element of the list must be hashable and part of the FlexDataset. Defaults to None.
            num_proc (int, optional): Number of processes to parallelize, negative values are ignored. Default to None (Use all).
            func (Callable, optional): Function to apply to preprocess the data. Defaults to None.

        Returns:
            FlexDataset: The modified FlexDataset.

        Raises:
            ValueError: If function is not given it raises an error.
        """
        if func is None:
            raise ValueError(
                "Function to apply can't be null. Please give a function to apply."
            )
        if num_proc is not None:
            num_proc = min(
                max(1, num_proc), len(self.keys())
            )  # do not allow negative num_proc
        if clients_ids is None:
            clients_ids = list(self.keys())
        elif any(client not in list(self.keys()) for client in clients_ids):
            raise ValueError("All client ids given must be in the FlexDataset.")

        new_fld = deepcopy(self)

        def clients_ids_iterable():
            for i in clients_ids:
                yield new_fld[i]

        with Pool(processes=num_proc) as p:
            chosen_clients = FlexDataset(
                {
                    client_id: result
                    for result, client_id in zip(
                        p.imap(  # We use imap because it is more memory efficient
                            partial(
                                func, *args, **kwargs
                            ),  # bind *args and **kwargs arguments to each call
                            clients_ids_iterable(),  # iterate over dict values only
                            chunksize=int(
                                num_proc or 1
                            ),  # 1 is the default value in case of None
                        ),
                        clients_ids,
                    )
                }
            )
        new_fld.update(chosen_clients)
        return new_fld

    def normalize(
        self,
        clients_ids: List[Hashable] = None,
        num_proc: int = None,
        *args,
        **kwargs,
    ):
        """Function that normalize the data over the clients.

        Args:
            fld (FlexDataset): FlexDataset containing all the data from the clients.
            clients_ids (List[Hashtable], optional): List containig the clients id whether
            to normalize the data or not. Each element of the list must be hashable. Defaults to None.
            num_proc (int, optional): Number of processes to paralelize. Default to None (Use all).

        Returns:
            FlexDataset: The FlexDataset normalized.
        """
        return self.map(clients_ids, num_proc, normalize, *args, **kwargs)

    def one_hot_encoding(
        self,
        clients_ids: List[Hashable] = None,
        num_proc: int = None,
        *args,
        **kwargs,
    ):
        """Function that apply one hot encoding to the client classes.

        Args:
            fld (FlexDataset): FlexDataset containing all the data from the clients.
            clients_ids (List[Hashtable], optional): List containing the clients id whether
            to normalize the data or not. Each element of the list must be hashable. Defaults to None.
            num_proc (int, optional): Number of processes to paralelize. Default to None (Use all).

        Returns:
            FlexDataset: The FlexDataset normalized.
        """
        return self.map(clients_ids, num_proc, one_hot_encoding, *args, **kwargs)
