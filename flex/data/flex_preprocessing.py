import multiprocessing
from typing import Callable

import numpy as np

from flex.data.flex_dataset import FlexDataset


def normalize(client):
    """Function that normalizes the data

    Args:
        X_data (_type_): _description_

    Returns:
        _type_: _description_
    """
    norm = np.linalg.norm(client.X_data, axis=0)
    if any(norm == 0):
        norm = np.finfo(client.X_data.dtype).eps
    return client.X_data / norm


def normalize_data_at_client(
    fld: FlexDataset, clients_ids: list = None, processes: int = None
):
    """Function that normalize the data over the clients.

    Args:
        fld (FlexDataset): FlexDataset containing all the data from the clients.
        clients_ids (list, optional): List containig the the clients id whether
        to normalize the data or not. Defaults to None.
        processes (int, optional): Number of processes to paralelize. Default to None (Use all).

    Returns:
        FlexDataset: The FlexDataset normalized.
    """
    if processes is not None:
        processes = min(processes, len(fld.keys()))
    pool = multiprocessing.Pool(processes=processes)
    if clients_ids is not None:
        chosen_clients = FlexDataset(
            {
                client_id: fld[client_id]
                for client_id in clients_ids
                if client_id in fld.keys()
            }
        )
    else:
        chosen_clients = fld
    # print(f"Client_1: {chosen_clients['client_1'].keys()}")
    print(f"Client_1: {chosen_clients['client_1'].X_data.shape}")
    processed_values = [
        pool.apply(normalize, args=client)  #  , kwds={"axis": axis})
        for _, client in chosen_clients.items()
    ]

    for (client_id, client_data), updated_X_data in zip(
        chosen_clients.items(), processed_values
    ):
        chosen_clients[client_id].X_data = updated_X_data.get(timeout=1)
    # chosen_clients = FlexDataset(dict(zip(chosen_clients.keys(), processed_values)))
    fld.update(chosen_clients)
    return fld


def preprocessing_custom_func(
    fld: FlexDataset,
    clients_ids: list = None,
    processes: int = None,
    func: Callable = None,
    *args,
    **kwargs,
):
    """This function applies a custom function to the FlexDataset in paralels.

    The *args and the **kwargs provided to this function are all the args and kwargs
    of the custom function provided by the client.

    Args:
        fld (FlexDataset): FlexDataset containing all the data from the clients.
        clients_ids (list, optional): List containig the the clients id whether
        to normalize the data or not. Defaults to None.
        processes (int, optional): Number of processes to paralelize. Default to None (Use all).
        func (Callable, optional): Function to apply to preprocess the data. Defaults to None.

    Returns:
        FlexDataset: The FlexDataset normalized.

    Raises:
        ValueError: If function is not given it raises an error.
    """
    if func is None:
        raise ValueError(
            "Function to apply can't be null. Please give a function to apply."
        )
    if processes is not None:
        processes = min(processes, len(fld.keys()))
    pool = multiprocessing.Pool(processes=processes)
    if clients_ids is not None:
        chosen_clients = FlexDataset(
            {
                client_id: fld[client_id]
                for client_id in clients_ids
                if client_id in fld.keys()
            }
        )
    else:
        chosen_clients = fld
    processed_values = [
        pool.apply(func=func, args=(client.X_data, args), kwds=kwargs)
        for client_id, client in chosen_clients.items()
    ]
    for (client_id, client_data), updated_X_data in zip(
        chosen_clients.items(), processed_values
    ):
        chosen_clients[client_id].X_data = updated_X_data
    # chosen_clients = FlexDataset(dict(zip(chosen_clients.keys(), processed_values)))
    fld.update(chosen_clients)
    return fld
