from copy import deepcopy

import numpy as np


def normalize(client, *args, **kwargs):
    """Function that normalizes the data

    Args:
        client (FlexDataObject): client whether to normalize the data.

    Returns:
        np.array: Returns data normalized
    """
    norms = np.linalg.norm(client.X_data, axis=1)
    norms = np.where(norms == 0, np.finfo(client.X_data.dtype).eps, norms)
    X_data = np.array([x / n for x, n in zip(client.X_data, norms)])
    new_client = deepcopy(client)
    new_client.X_data = X_data
    return new_client


def one_hot_encoding(client, *args, **kwargs):
    """Function that apply one hot encoding to the labels of a client.

    Args:
        client (FlexDataObject): client wheter to one hot encode his classes.

    Raises:
        ValueError: Raises value error if n_classes is not given in the kwargs argument.

    Returns:
        FlexDataObject: Returns the client with the y_data property updated.
    """
    if "n_classes" not in kwargs:
        raise ValueError(
            "No number of classes given. The parameter n_classes must be given through kwargs."
        )
    n_classes = int(kwargs["n_classes"])
    one_hot_classes = np.zeros((client.y_data.size, n_classes))
    one_hot_classes[np.arange(client.y_data.size), client.y_data] = 1
    new_client = deepcopy(client)
    new_client.y_data = one_hot_classes
    return new_client
