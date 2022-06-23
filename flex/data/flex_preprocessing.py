from typing import Callable

import numpy as np

from flex.data.flex_dataset import FlexDataset


def normalize(client, *args, **kwargs):
    """Function that normalizes the data

    Args:
        client (FlexDataObject): client whether to normalize the data.

    Returns:
        np.array: Returns data normalized
    """
    norm = np.linalg.norm(client.X_data, axis=0)
    if any(norm == 0):
        norm = np.finfo(client.X_data.dtype).eps
    client.X_data = client.X_data / norm
    return client


def normalize_data_at_client(
    fld: FlexDataset,
    clients_ids: list = None,
    processes: int = None,
    *args,
    **kwargs,
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
    return preprocessing_custom_func(
        fld, clients_ids, processes, normalize, *args, **kwargs
    )


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
    if clients_ids is None:
        clients_ids = list(fld.keys())
    chosen_clients = FlexDataset(
        {client_id: func(fld[client_id], *args, **kwargs) for client_id in clients_ids}
    )
    fld.update(chosen_clients)
    return fld
