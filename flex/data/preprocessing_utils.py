from copy import deepcopy

import numpy as np

from flex.data.dataset import Dataset


def normalize(client, *args, **kwargs):
    """Function that normalizes the federated data.

    Args:
        client (Dataset): client whether to normalize the data.

    Returns:
        Dataset: Returns the client with the X_data property normalized.
    """
    norms = np.linalg.norm(client.X_data, axis=0)
    norms = np.where(norms == 0, np.finfo(client.X_data.dtype).eps, norms)
    new_X_data = deepcopy(client.X_data) / norms
    return Dataset(X_data=new_X_data, y_data=deepcopy(client.y_data))


def one_hot_encoding(client, *args, **kwargs):
    """Function that apply one hot encoding to the labels of a client.

    Args:
        client (Dataset): client to which apply one hot encode to her classes.

    Raises:
        ValueError: Raises value error if n_classes is not given in the kwargs argument.

    Returns:
        Dataset: Returns the client with the y_data property updated.
    """
    if "n_classes" not in kwargs:
        raise ValueError(
            "No number of classes given. The parameter n_classes must be given through kwargs."
        )
    n_classes = int(kwargs["n_classes"])
    one_hot_classes = np.zeros((client.y_data.size, n_classes))
    one_hot_classes[np.arange(client.y_data.size), client.y_data] = 1
    new_client_y_data = one_hot_classes
    return Dataset(X_data=deepcopy(client.X_data), y_data=new_client_y_data)
