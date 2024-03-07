"""
Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
"""File that contains some utils functions to test help us testing the datasets that works with FLEXible.
"""
import gc  # noqa: E402

from flex.data import Dataset, FedDataDistribution, FedDatasetConfig  # noqa: E402

config = FedDatasetConfig(
    seed=0,
    n_nodes=2,
    replacement=False,
    node_ids=["client_0", "client_1"],
)


def iterate_module_functions(module):
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
            fld = Dataset.from_torchtext_dataset(data)
            fld.validate()
            flex_dataset = FedDataDistribution.from_pytorch_text_dataset(data, config)
            del flex_dataset
            flex_dataset = FedDataDistribution.from_config(fld, config)
            del flex_dataset
            valid_datasets.append([name, func, "-"])
            del data
            del fld
        except Exception as e:
            wrong_datasets.append([name, func, e])
    return valid_datasets, wrong_datasets


def check_if_can_load_hf_dataset():
    """Function that takes a list of huggingface datasets and check whether
    or not can be loaded to FLEXible with the methods available. For those
    that gives error, we keep the error. Also, the user may indicate a list of
    columns that will be the features of the model (X_column) and the columns
    that will be the label (y_column) of the model.

    To check the datasets, refer to PluggableDatasetsHuggingFace

    Returns:
        list, list: The first list contains the names of each database that
        can be loaded into FLEXible. The second list contains the names of
        each database that gives error while trying to load it to FLEXible.

    Raises:
        - Gives error if the database can't be loaded to FLEXible.
        - Gives error if the database can't be loaded as: func(split='train')
    """
    from datasets import load_dataset

    from flex.data.pluggable_datasets import PluggableHuggingFace

    valid_datasets = []
    wrong_datasets = []
    for dataset in PluggableHuggingFace:
        name, X_columns, y_column = dataset.value
        print(f"Testing dataset: {name}")
        try:
            data = load_dataset(name, split="train")
            flex_dataset = FedDataDistribution.from_config_with_huggingface_dataset(
                data, config, X_columns, y_column
            )
            del flex_dataset
            fld = Dataset.from_huggingface_dataset(data, X_columns, y_column)
            fld.validate()
            flex_dataset = FedDataDistribution.from_config(fld, config)
            del flex_dataset
            del data
            del fld
            valid_datasets.append([name, "-"])
        except Exception as e:
            wrong_datasets.append([name, e])
    return valid_datasets, wrong_datasets


def check_if_can_load_text_tfds():
    """Function that take a list of tensorflow datasets and check whether
    or not can be loaded to FLEXible with the methods available. For those
    that gives error, we keep the error. Also, the user may indicate a list of
    columns that will be the features of the model (X_column) and the columns
    that will be the label (y_column) of the model.

    To check the datasets refer to PluggableDatasetsTensorFlowText

    Returns:
        list, list: The first list contains the names of each database that
        can be loaded into FLEXible. The second list contains the names of
        each database that gives error while trying to load it to FLEXible.

    Raises:
        - Gives error if the database can't be loaded to FLEXible.
        - Gives error with Question Answering datasets. Probably because
        they needs further preprocessing than other tasks.
            -> Example of an error:  AttributeError("'dict' object has no attribute 'shape'")
    """
    import tensorflow_datasets as tfds

    from flex.data import PluggableDatasetsTensorFlowText

    valid_datasets = []
    wrong_datasets = []
    for dataset in PluggableDatasetsTensorFlowText:
        name, X_columns, y_column = dataset.value
        print(f"Testing dataset: {name}")
        try:
            split = "train" if name != "asset" else "validation"
            data = tfds.load(name, split=split)
            flex_dataset = FedDataDistribution.from_config_with_tfds_dataset_args(
                data, config, X_columns, y_column
            )
            fld = Dataset.from_tfds_dataset_with_args(data, X_columns, y_column)
            fld.validate()
            flex_dataset = FedDataDistribution.from_config(fld, config)
            del flex_dataset
            del data
            del fld
            valid_datasets.append([name, "-"])
        except Exception as e:
            wrong_datasets.append([name, e])
    return valid_datasets, wrong_datasets


def check_if_can_load_torchvision_dataset(list_datasets, **kwargs):
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
            data = func(**kwargs)
            fld = Dataset.from_torchvision_dataset(data)
            fld.validate()
            flex_dataset = FedDataDistribution.from_config(fld, config)
            del flex_dataset
            gc.collect()
            flex_dataset = FedDataDistribution.from_config_with_torchvision_dataset(
                data, config
            )
            del flex_dataset
            del data
            del fld
            valid_datasets.append([name, func, "-"])
        except Exception as e:
            wrong_datasets.append([name, func, e])
        gc.collect()
    return valid_datasets, wrong_datasets
