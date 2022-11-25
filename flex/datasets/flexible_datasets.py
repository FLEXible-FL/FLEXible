from abc import ABC, abstractmethod
from importlib import import_module
from inspect import isfunction


def list_datasets():
    """
    Function that returns the available datasets in FLEXible.
    Returns: A list with the available datasets.
    -------

    """
    return list(FLEXibleDatasets.instances.keys())


def check_availability(name, func):
    """
    Function that checks if a function is a dataset or not. Here we have a black list of functions,
    for those functions that are imports for another modules, so we ignore them.

    Parameters
    ----------
    name: str -> Name of the function
    func: callable -> Function to check

    Returns: True/False if the function is a function and is not in the black list
    -------

    """
    black_list = ['loadmat', 'download_dataset']
    return bool(isfunction(func) and name not in black_list)


class FLEXibleDatasets:
    """
    Class that help to load every dataset available in FLEXible. Those that can be load as a federated dataset,
    will contain the word 'federated' in their name and will be loaded as FedDataset (flex.data.FedDataset).
    The rest datasets are standard datasets, and will be loaded as Dataset (flex.data.Dataset).

    This class can't be instantiated. To load a dataset use the load function.
    """
    __module_names = ['flex.datasets.federated_datasets', 'flex.datasets.standard_datasets']
    instances = {str(name).lower(): val for module in __module_names for name, val in import_module(module).__dict__.items() if check_availability(name, val)}
    __create_key = object()

    def __init__(self, create_key: object = None) -> None:
        assert (
                create_key == FLEXibleDatasets.__create_key
        ), """Datasets objects must be created using Datasets.load"""


def load(name: str = '.', out_dir: str = ".", split=None, return_test: bool = None, **kwargs):
    """
    Function that loads every
    Parameters
    ----------
    name
    out_dir
    split
    return_test
    kwargs

    Returns
    -------

    """
    if name.lower() not in FLEXibleDatasets.instances:
        raise ValueError(
            f"Database {name} is not available. Please, check the available datasets using the list_datasets "
            f"function. "
        )
    return FLEXibleDatasets.instances[name](out_dir=out_dir, split=split, return_test=return_test, **kwargs)
