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
from collections import UserDict
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Hashable, List, Optional

from multiprocess import Pool

from flex.common.utils import check_min_arguments
from flex.data.dataset import Dataset
from flex.data.preprocessing_utils import normalize, one_hot_encoding


class FedDataset(UserDict):
    """Class that represents a federated dataset for the Flex library.
    The dataset contains the ids of the nodes and the dataset associated
    with each node.

    Attributes
    ----------
    data (collections.UserDict): The structure is a dictionary
        with the node ids as keys and the dataset as value.
    """

    def __setitem__(self, key: Hashable, item: Dataset) -> None:
        self.data[key] = item

    def get(self, key: Hashable, default: Optional[Any] = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def apply(
        self,
        func: Callable,
        node_ids: List[Hashable] = None,
        num_proc: int = 1,
        **kwargs,
    ):
        r"""This function lets apply a custom function to the FlexDataset in parallel.

        The \**kwargs provided to this function are all the kwargs of the custom function provided by the node.

        Args:
        -----
            func (Callable, optional): Function to apply to preprocess the data.
            node_ids (List[Hashtable], optional): List containig the the node ids where func will
            be applied. Each element of the list must be hashable and part of the FlexDataset. Defaults to None.
            num_proc (int, optional): Number of processes to parallelize, negative values are ignored. Default to 1

        Returns:
        --------
            FedDataset: The modified FlexDataset.

        Raises:
        -------
            ValueError: All node ids given must be in the FlexDataset.

        """

        if node_ids is None:
            node_ids = list(self.keys())
        elif isinstance(node_ids, str):
            if node_ids not in self.keys():
                raise ValueError("All node ids given must be in the FedDataset.")
        elif any(node not in self.keys() for node in node_ids):
            raise ValueError("All nodes ids given must be in the FedDataset.")

        error_msg = f"The provided function: {func.__name__} is expected to have at least 1 argument/s."
        assert check_min_arguments(func, min_args=1), error_msg

        # if any(self[i].X_data._is_generator for i in node_ids):
        #     raise NotImplementedError("LazyIndexable with generators will be supported soon")

        if num_proc < 2:
            updates = self._map_single(func, node_ids, **kwargs)
        else:
            f = partial(self._map_single, func)
            updates = self._map_parallel(f, node_ids, num_proc=num_proc, **kwargs)

        new_fld = deepcopy(
            self
        )  # seguramente solo haga falta copiar los que no están en node_ids
        new_fld.update(updates)
        return new_fld

    def _map_parallel(
        self,
        func: Callable,
        node_ids: List[Hashable],
        num_proc: int = 2,
        **kwargs,
    ):
        """This function lets apply a custom function to the FlexDataset in parallel.

        The  **kwargs provided to this function are the kwargs of the custom function provided by the node.

        Args:
        -----
            fld (FedDataset): FlexDataset containing all the data from the nodes.
            func (Callable): Function to apply to preprocess the data.
            node_ids (List[Hashtable]): List containig the the nodes id where func will
            be applied. Each element of the list must be hashable and part of the FlexDataset
            num_proc (int): Number of processes to parallelize, negative values are ignored. Default to 2

        Returns:
        --------
            FedDataset: The modified FlexDataset.

        """

        updates = {}
        f = partial(func, **kwargs)  # bind **kwargs arguments to each call

        with Pool(processes=num_proc) as p:
            for i in p.imap(f, node_ids):
                updates.update(i)

        return updates

    def _map_single(
        self,
        func: Callable,
        node_ids: List[Hashable],
        **kwargs,
    ):
        """This function lets apply a custom function to the FlexDataset secuentially.

        This functions will be used by default in the map function, because of the error
        generated by a bug with the multiprocessing library. If you want to check the error
        to try to use the _map_parallel

        The *args and the **kwargs provided to this function are all the args and kwargs
        of the custom function provided by the node.

        Args:
        -----
            func (Callable): Function to apply to preprocess the data.
            node_ids (List[Hashtable]): List containig the the node ids where func will
            be applied. Each element of the list must be hashable and part of the FlexDataset.

        Returns:
        --------
            FedDataset: The modified FlexDataset.
        """
        if not isinstance(node_ids, list):
            node_ids = [node_ids]

        return {node_id: func(self[node_id], **kwargs) for node_id in node_ids}

    def normalize(
        self,
        node_ids: List[Hashable] = None,
        num_proc: int = 0,
        *args,
        **kwargs,
    ):
        """Function that normalize the data over the nodes.

        Args:
        -----
            fld (FedDataset): FlexDataset containing all the data from the nodes.
            node_ids (List[Hashtable], optional): List containig the nodes id whether
            to normalize the data or not. Each element of the list must be hashable. Defaults to None.
            num_proc (int, optional): Number of processes to paralelize. Default to None (Use all).

        Returns:
        --------
            FedDataset: The FlexDataset normalized.
        """
        return self.apply(normalize, node_ids, num_proc, *args, **kwargs)

    def one_hot_encoding(
        self,
        node_ids: List[Hashable] = None,
        num_proc: int = 0,
        *args,
        **kwargs,
    ):
        """Function that apply one hot encoding to the node labels.

        Args:
        -----
            fld (FedDataset): FlexDataset containing all the data from the nodes.
            node_ids (List[Hashtable], optional): List containing the nodes id whether
            to normalize the data or not. Each element of the list must be hashable. Defaults to None.
            num_proc (int, optional): Number of processes to paralelize. Default to None (Use all).

        Returns:
        --------
            FedDataset: The FlexDataset normalized.
        """
        return self.apply(one_hot_encoding, node_ids, num_proc, *args, **kwargs)
