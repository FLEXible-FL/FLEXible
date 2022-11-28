from inspect import getmembers, isfunction

from flex.datasets import federated_datasets, standard_datasets


def list_datasets():
    """
    Function that returns the available datasets in FLEXible.
    Returns: A list with the available datasets.
    -------

    """
    return list(ds_invocation.keys())


def load(name, *args, **kwargs):
    """
    Function that loads every dataset builtin into the framework
    Parameters
    ----------
    name


    Returns
    -------

    """
    try:
        caller = ds_invocation[name.lower()]
    except KeyError as e:
        raise ValueError(
            f"{name} is not available. Please, check available datasets using list_datasets "
            f"function. "
        ) from e

    return caller(*args, **kwargs)


ds_invocation = dict(
    getmembers(standard_datasets, isfunction)
    + getmembers(federated_datasets, isfunction)
)
