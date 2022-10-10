"""File that contains some utils functions to test help us testing the datasets that works with FLEXible.
"""

from flex.data import FlexDataDistribution, FlexDataObject, FlexDatasetConfig

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


def check_if_can_load_torchtext_dataset(list_datasets):
    """Function that recieve a list of torchtext datasets and check whether
    or not can be loaded to FLEXible with the methods available. For those
    that gives error, we keep the error.

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
            data = func(split="train")
            fld = FlexDataObject.from_torchtext_dataset(data)
            fld.validate()
            flex_dataset = FlexDataDistribution.from_pytorch_text_dataset(data, config)
            del flex_dataset
            flex_dataset = FlexDataDistribution.from_config(fld, config)
            del flex_dataset
            valid_datasets.append([name, func, "-"])
            del data
            del fld
        except Exception as e:
            wrong_datasets.append([name, func, e])
    return valid_datasets, wrong_datasets


def check_if_can_load_hf_dataset():
    """Function that recieve a list of huggingface datasets and check whether
    or not can be loaded to FLEXible with the methods available. For those
    that gives error, we keep the error. Also, the user may indicate a list of
    columns that will be the features of the model (X_column) and the columns
    that will be the label (y_column) of the model.

    Args:
        list_datasets (list): List of strings containing the names of the datasets
        that will be tested.
        list_X_columns (List[Union[List, String]]): List where each element is
        a list or a string and represents the features that will be used in the
        model.
        list_y_columns (List[Union[List, String]]): List where each element is
        a list or a string and represents the labels that will be used in the
        model.

    Returns:
        list, list: The first list contains the names of each database that
        can be loaded into FLEXible. The second list contains the names of
        each database that gives error while trying to load it to FLEXible.

    Raises:
        - Gives error if the database can't be loaded to FLEXible.
        - Gives error if the database can't be loaded as: func(split='train')
    """
    from datasets import load_dataset

    from flex.data.pluggable_datasets import PluggableDatasetsHuggingFace

    valid_datasets = []
    wrong_datasets = []
    for dataset in PluggableDatasetsHuggingFace:
        name, X_columns, y_column = dataset.value
        print(f"Testing dataset: {name}")
        try:
            data = load_dataset(name, split="train")
            flex_dataset = FlexDataDistribution.from_config_with_huggingface_dataset(
                data, config, X_columns, y_column
            )
            del flex_dataset
            fld = FlexDataObject.from_huggingface_dataset(data, X_columns, y_column)
            fld.validate()
            flex_dataset = FlexDataDistribution.from_config(fld, config)
            del flex_dataset
            del data
            del fld
            valid_datasets.append([name, "-"])
        except Exception as e:
            wrong_datasets.append([name, e])
    return valid_datasets, wrong_datasets
