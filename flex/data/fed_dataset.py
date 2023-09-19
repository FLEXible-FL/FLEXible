# Copyright 2023 Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


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
    The dataset contains the ids of the clients and the dataset associated
    with each client.

    Attributes
    ----------
    data (collections.UserDict): The structure is a dictionary
        with the clients ids as keys and the dataset as value.
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
        clients_ids: List[Hashable] = None,
        num_proc: int = 1,
        **kwargs,
    ):
        """This function lets apply a custom function to the FlexDataset in parallel.

        The **kwargs provided to this function are all the kwargs of the custom function provided by the client.

        Args:
            func (Callable, optional): Function to apply to preprocess the data.
            clients_ids (List[Hashtable], optional): List containig the the clients id where func will
            be applied. Each element of the list must be hashable and part of the FlexDataset. Defaults to None.
            num_proc (int, optional): Number of processes to parallelize, negative values are ignored. Default to 1

        Returns:
            FedDataset: The modified FlexDataset.

        Raises:
            ValueError: All client ids given must be in the FlexDataset.

        """

        if clients_ids is None:
            clients_ids = list(self.keys())
        elif isinstance(clients_ids, str):
            if clients_ids not in self.keys():
                raise ValueError("All client ids given must be in the FedDataset.")
        elif any(client not in self.keys() for client in clients_ids):
            raise ValueError("All client ids given must be in the FedDataset.")

        error_msg = f"The provided function: {func.__name__} is expected to have at least 1 argument/s."
        assert check_min_arguments(func, min_args=1), error_msg

        # if any(self[i].X_data._is_generator for i in clients_ids):
        #     raise NotImplementedError("LazyIndexable with generators will be supported soon")

        if num_proc < 2:
            updates = self._map_single(func, clients_ids, **kwargs)
        else:
            f = partial(self._map_single, func)
            updates = self._map_parallel(f, clients_ids, num_proc=num_proc, **kwargs)

        new_fld = deepcopy(
            self
        )  # seguramente solo haga falta copiar los que no están en clients_ids
        new_fld.update(updates)
        return new_fld

    def _map_parallel(
        self,
        func: Callable,
        clients_ids: List[Hashable],
        num_proc: int = 2,
        **kwargs,
    ):
        """This function lets apply a custom function to the FlexDataset in parallel.

        The  **kwargs provided to this function are the kwargs of the custom function provided by the client.

        Args:
            fld (FedDataset): FlexDataset containing all the data from the clients.
            func (Callable): Function to apply to preprocess the data.
            clients_ids (List[Hashtable]): List containig the the clients id where func will
            be applied. Each element of the list must be hashable and part of the FlexDataset
            num_proc (int): Number of processes to parallelize, negative values are ignored. Default to 2

        Returns:
            FedDataset: The modified FlexDataset.

        """

        updates = {}
        f = partial(func, **kwargs)  # bind **kwargs arguments to each call

        with Pool(processes=num_proc) as p:
            for i in p.imap(f, clients_ids):
                updates.update(i)

        return updates

    def _map_single(
        self,
        func: Callable,
        clients_ids: List[Hashable],
        **kwargs,
    ):
        """This function lets apply a custom function to the FlexDataset secuentially.

        This functions will be used by default in the map function, because of the error
        generated by a bug with the multiprocessing library. If you want to check the error
        to try to use the _map_parallel

        The *args and the **kwargs provided to this function are all the args and kwargs
        of the custom function provided by the client.

        Args:
            func (Callable): Function to apply to preprocess the data.
            clients_ids (List[Hashtable]): List containig the the clients id where func will
            be applied. Each element of the list must be hashable and part of the FlexDataset.

        Returns:
            FedDataset: The modified FlexDataset.
        """
        if not isinstance(clients_ids, list):
            clients_ids = [clients_ids]

        return {client_id: func(self[client_id], **kwargs) for client_id in clients_ids}

    def normalize(
        self,
        clients_ids: List[Hashable] = None,
        num_proc: int = 0,
        *args,
        **kwargs,
    ):
        """Function that normalize the data over the clients.

        Args:
            fld (FedDataset): FlexDataset containing all the data from the clients.
            clients_ids (List[Hashtable], optional): List containig the clients id whether
            to normalize the data or not. Each element of the list must be hashable. Defaults to None.
            num_proc (int, optional): Number of processes to paralelize. Default to None (Use all).

        Returns:
            FedDataset: The FlexDataset normalized.
        """
        return self.apply(normalize, clients_ids, num_proc, *args, **kwargs)

    def one_hot_encoding(
        self,
        clients_ids: List[Hashable] = None,
        num_proc: int = 0,
        *args,
        **kwargs,
    ):
        """Function that apply one hot encoding to the client classes.

        Args:
            fld (FedDataset): FlexDataset containing all the data from the clients.
            clients_ids (List[Hashtable], optional): List containing the clients id whether
            to normalize the data or not. Each element of the list must be hashable. Defaults to None.
            num_proc (int, optional): Number of processes to paralelize. Default to None (Use all).

        Returns:
            FedDataset: The FlexDataset normalized.
        """
        return self.apply(one_hot_encoding, clients_ids, num_proc, *args, **kwargs)
