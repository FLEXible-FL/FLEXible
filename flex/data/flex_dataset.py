from collections import UserDict
from copy import deepcopy
from functools import partial
from multiprocessing import Pool
from typing import Any, Callable, Hashable, List, Optional

from flex.data.flex_data_object import FlexDataObject
from flex.data.flex_preprocessing_utils import normalize, one_hot_encoding


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
        """This function lets apply a custom function to the FlexDataset in parallel.

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
            AttributeError: The multriprocessing function has a known bug whether you have to create a function in other file
            in orther to apply it in parallel. If this is not done, the map function will keep ciclying infinitely
            and will not end. To solve this error, and for testing purpose for custom functions, we highly
            recommend the use of num_proc to None, to use the map function in secuentialy. To improve the
            performance of the custom function, the user will hace to create the function in another file and
            import it in orther to use it. We found and example of the error we're talking about in Stack Overflow,
            be sure to check it: https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror/42383397
            Also, here is the link to the bug: https://bugs.python.org/issue25053

        """
        if func is None:
            raise ValueError(
                "Function to apply can't be null. Please give a function to apply."
            )

        if clients_ids is None:
            clients_ids = list(self.keys())
        elif any(client not in list(self.keys()) for client in clients_ids):
            raise ValueError("All client ids given must be in the FlexDataset.")

        if num_proc is None:
            # Execute without using multiprocessing to avoid bugs from multiprocssing
            return self._map_single(clients_ids, func, *args, **kwargs)

        num_proc = min(max(1, num_proc), len(self.keys()))
        return self._map_parallel(clients_ids, num_proc, func, *args, **kwargs)

    def _map_single(
        self,
        clients_ids: List[Hashable],
        func: Callable,
        *args,
        **kwargs,
    ):
        """This function lets apply a custom function to the FlexDataset secuentially.

        This functions will be used by default in the map function, because of the error
        generated by a bug with the multiprocessing library. If you want to check the error
        to try to use the _map_parallel

        The *args and the **kwargs provided to this function are all the args and kwargs
        of the custom function provided by the client.

        Args:
            fld (FlexDataset): FlexDataset containing all the data from the clients.
            clients_ids (List[Hashtable]): List containig the the clients id where func will
            be applied. Each element of the list must be hashable and part of the FlexDataset. Defaults to None.
            func (Callable): Function to apply to preprocess the data. Defaults to None.

        Returns:
            FlexDataset: The modified FlexDataset.
        """
        new_fld = deepcopy(self)
        chosen_clients = FlexDataset(
            {
                client_id: func(new_fld[client_id], *args, **kwargs)
                for client_id in clients_ids
            }
        )
        new_fld.update(chosen_clients)
        return new_fld

    def _map_parallel(
        self,
        clients_ids: List[Hashable],
        num_proc: int,
        func: Callable,
        *args,
        **kwargs,
    ):
        """This function lets apply a custom function to the FlexDataset in parallel.

        The *args and the **kwargs provided to this function are all the args and kwargs
        of the custom function provided by the client.

        Args:
            fld (FlexDataset): FlexDataset containing all the data from the clients.
            clients_ids (List[Hashtable]): List containig the the clients id where func will
            be applied. Each element of the list must be hashable and part of the FlexDataset. Defaults to None.
            num_proc (int): Number of processes to parallelize, negative values are ignored. Default to None (Use all).
            func (Callable): Function to apply to preprocess the data. Defaults to None.

        Returns:
            FlexDataset: The modified FlexDataset.

        Raises:
            ValueError: If function is not given it raises an error.
            AttributeError: The multriprocessing function has a known bug whether you have to create a function in other file
            in orther to apply it in parallel. If this is not done, the map function will keep ciclying infinitely
            and will not end. To solve this error, and for testing purpose for custom functions, we highly
            recommend the use of num_proc to None, to use the map function in secuentialy. To improve the
            performance of the custom function, the user will hace to create the function in another file and
            import it in orther to use it. We found and example of the error we're talking about in Stack Overflow,
            be sure to check it: https://stackoverflow.com/questions/41385708/multiprocessing-example-giving-attributeerror/42383397
            Also, here is the link to the bug: https://bugs.python.org/issue25053
        """
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
