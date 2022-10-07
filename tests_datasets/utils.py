"""File that contains some utils functions to test help us testing the datasets that works with FLEXible.
"""

from operator import mod
from flex.data import FlexDataObject, FlexDataDistribution, FlexDatasetConfig

config = FlexDatasetConfig(
            seed=0,
            n_clients=2,
            replacement=False,
            client_names=["client_0", "client_1"],
        )


def iterate_module_functions_torchtext(module):
    """Function to get the functions that load a torchtext dataset.

    Args:
        module (module): torchtext.datasets

    Returns:
        List: A list that contains the name and the function of the 
        datasets available in torchtext.datasets.
    """
    return [[name, val] for name, val in module.__dict__.items() if callable(val)]

def check_if_can_load_torch_dataset(list_datasets):
    """Function that recieve a list of datasets and check whether or not
    can be loaded to FLEXible with the methods available. For those that
    gives error, we keep the error.

    Args:
        list_datasets (list): List of list (name, func) containing the datasets
        that will be tested.

    Returns:
        list, list: The first list contains the names of each database that
        can be loaded into FLEXible. The second list contains the names of
        each database that gives error while trying to load it to FLEXible.

    Raises:
        - Gives error if the database can't be loaded to FLEXible.
        - Gives error if the database can't be loaded as: func(split='train')
    """
    valid_datasets = []
    wrong_datasets = []
    for name, func in list_datasets:
        print(f"Testing dataset: {name}")
        try:
            data = func(split='train')
            fld = FlexDataObject.from_torchtext_dataset(data)
            fld.validate()
            flex_dataset = FlexDataDistribution.from_pytorch_text_dataset(data, config)
            flex_dataset = FlexDataDistribution.from_config(fld, config)
            valid_datasets.append([name, func, '-'])
        except Exception as e:
            wrong_datasets.append([name, func, e])
    return valid_datasets, wrong_datasets
