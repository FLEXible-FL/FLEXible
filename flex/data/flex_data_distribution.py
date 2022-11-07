import copy
from collections import defaultdict
from math import floor
from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt
from numpy.random import default_rng
from scipy.io import loadmat

from flex.data import FlexDataObject, FlexDataset, FlexDatasetConfig
from flex.data.flex_utils import (
    MNIST_DIGITS,
    MNIST_FILE,
    MNIST_MD5,
    MNIST_URL,
    download_dataset,
)


class FlexDataDistribution(object):
    __create_key = object()

    def __init__(self, create_key: object = None) -> None:
        assert (
            create_key == FlexDataDistribution.__create_key
        ), """FlexDataDistribution objects must be created using FlexDataDistribution.from_config or
        FlexDataDistribution.iid_distribution"""

    @classmethod
    def MNIST(cls, out_dir: str = ".", include_writers=False):
        mnist_files = download_dataset(
            MNIST_URL, MNIST_FILE, MNIST_MD5, extract=True, output=True
        )
        dataset = [
            loadmat(mat)["dataset"] for mat in mnist_files if MNIST_DIGITS in mat
        ][0]
        writers = dataset["train"][0, 0]["writers"][0, 0]
        train_data = np.reshape(
            dataset["train"][0, 0]["images"][0, 0], (-1, 28, 28), order="F"
        )
        train_labels = np.squeeze(dataset["train"][0, 0]["labels"][0, 0])
        if include_writers:
            train_data = [(v, writers[i][0]) for i, v in enumerate(train_data)]

        test_data = np.reshape(
            dataset["test"][0, 0]["images"][0, 0], (-1, 28, 28), order="F"
        )
        test_labels = np.squeeze(dataset["test"][0, 0]["labels"][0, 0])

        train_data_object = FlexDataObject(
            X_data=np.asarray(train_data), y_data=train_labels
        )
        test_data_object = FlexDataObject(
            X_data=np.asarray(test_data), y_data=test_labels
        )
        return train_data_object, test_data_object

    @classmethod
    def FederatedMNIST(cls, out_dir: str = ".", return_test=False):
        train_data, test_data = cls.MNIST(out_dir, include_writers=True)
        config = FlexDatasetConfig(group_by_feature=1)
        federated_data = cls.from_config(train_data, config)
        if return_test:
            return federated_data, test_data
        else:
            return federated_data

    @classmethod
    def from_config_with_torchtext_dataset(cls, data, config: FlexDatasetConfig):
        """This function federates a centralized torchtext dataset given a FlexDatasetConfig.
        This function will transform the torchtext dataset into a FlexDataObject and then it will
        federate it.

        Args:
            data (Dataset): The torchtext dataset
            config (FlexDatasetConfig): FlexDatasetConfig with the configuration to federate the centralized dataset.
        """
        cdata = FlexDataObject.from_torchtext_dataset(data)
        return cls.from_config(cdata, config)

    @classmethod
    def from_config_with_tfds_image_dataset(cls, data, config: FlexDatasetConfig):
        """This function federates a centralized tensorflow dataset given a FlexDatasetConfig.
        This function will transform a dataset from the tensorflow_datasets module into a FlexDataObject
        and then it will federate it.

        Args:
            data (Dataset): The tensorflow dataset
            config (FlexDatasetConfig): FlexDatasetConfig with the configuration to federate the centralized dataset.
        """
        cdata = FlexDataObject.from_tfds_image_dataset(data)
        return cls.from_config(cdata, config)

    @classmethod
    def from_config_with_tfds_text_dataset(
        cls, data, config: FlexDatasetConfig, X_columns: list, label_column: list
    ):
        """This function federates a centralized tensorflow dataset given a FlexDatasetConfig.
        This function will transform a dataset from the tensorflow_datasets module into a FlexDataObject
        and then it will federate it.

        Args:
            data (Dataset): The tensorflow dataset
            config (FlexDatasetConfig): FlexDatasetConfig with the configuration to federate the centralized dataset.
            X_columns (List): List that contains the columns names for the input features.
            label_column (List): List that contains the columns names for the output features.
        """
        cdata = FlexDataObject.from_tfds_text_dataset(data, X_columns, label_column)
        return cls.from_config(cdata, config)

    @classmethod
    def from_config_with_torchvision_dataset(cls, data, config: FlexDatasetConfig):
        """This function federates a centralized torchvision dataset given a FlexDatasetConfig.
        This function will transform a dataset from the torchvision module into a FlexDataObject
        and then it will federate it.

        Args:
            data (Dataset): The torchvision dataset
            config (FlexDatasetConfig): FlexDatasetConfig with the configuration to federate the centralized dataset.
        """
        cdata = FlexDataObject.from_torchvision_dataset(data)
        return cls.from_config(cdata, config)

    @classmethod
    def from_config_with_huggingface_dataset(
        cls,
        data,
        config: FlexDatasetConfig,
        X_columns: list,
        label_column: str,
    ):
        """This function federates a centralized hugginface dataset given a FlexDatasetConfig.
        This function will transform a dataset from the HuggingFace Hub datasets into a FlexDataObject
        and then it will federate it.

        Args:
            data (Dataset): The hugginface dataset
            config (FlexDatasetConfig): FlexDatasetConfig with the configuration to federate the centralized dataset.
        """
        cdata = FlexDataObject.from_huggingface_dataset(data, X_columns, label_column)
        return cls.from_config(cdata, config)

    @classmethod
    def from_clustering_func(cls, cdata: FlexDataObject, clustering_func: Callable):
        """This function federates data into clients by means of a clustering function, that outputs
        to which client (cluster) a data point belongs.

        Args:
            cdata (FlexDataObject): Centralized dataset represented as a FlexDataObject.
            clustering_func (Callable): function that receives as arguments a pair of x and y elements from cdata
            and returns the name of the client (cluster) that should own it, the returned type must be Hashable.
            Note that we only support one client (cluster) per data point.

        Returns:
            federated_dataset (FlexDataset): The federated dataset.
        """
        d = defaultdict(list)
        for idx, (x, y) in enumerate(cdata):
            client_name = clustering_func(x, y)
            d[client_name].append(idx)

        config = FlexDatasetConfig(
            n_clients=len(d),
            client_names=list(d.keys()),
            indexes_per_client=list(d.values()),
            replacement=False,
        )
        return cls.from_config(cdata, config)

    @classmethod
    def iid_distribution(cls, cdata: FlexDataObject, n_clients: int = 2):
        """Function to create a FlexDataset for an IID experiment. We consider the simplest situation
        in which the data is distributed by giving the same amount of data to each client.

        Args:
            cdata (FlexDataObject): Centralized dataset represented as a FlexDataObject.
            n_clients (int): Number of clients in the Federated Learning experiment. Default 2.

        Returns:
            federated_dataset (FlexDataset): The federated dataset.
        """
        config = FlexDatasetConfig(n_clients=n_clients)
        return FlexDataDistribution.from_config(cdata, config)

    @classmethod
    def from_config(cls, cdata: FlexDataObject, config: FlexDatasetConfig):
        """This function prepare the data from a centralized data structure to a federated one.
        It will run different modifications to federate the data.

        Args:
            cdata (FlexDataObject): Centralized dataset represented as a FlexDataObject.
            config (FlexDatasetConfig): FlexDatasetConfig with the configuration to federate the centralized dataset.

        Returns:
            federated_dataset (FlexDataset): The federated dataset.
        """
        cdata.validate()
        config.validate()
        rng = default_rng(seed=config.seed)

        config_ = copy.deepcopy(config)  # copy, because we might modify some components

        if config.client_names is None:
            config_.client_names = list(range(config_.n_clients))

        # Normalize weights when no replacement
        if (
            not config_.replacement
            and config_.weights is not None
            and sum(config_.weights) > 1
        ):
            config_.weights = np.array(
                [w / sum(config.weights) for w in config.weights]
            )
        # Ensure that classes_per_client is translated to weights_per_class
        if config_.classes_per_client:
            cls.__configure_weights_per_class(rng, config_, cdata)
        # Normalize weights_per_class when no replacement
        if (
            not config_.replacement
            and config_.weights_per_class is not None
            and any(np.sum(config_.weights_per_class, axis=0) > 1)
        ):
            with np.errstate(divide="ignore", invalid="ignore"):  # raise no warnings
                config_.weights_per_class = config_.weights_per_class / np.sum(
                    config_.weights_per_class, axis=0
                )
            # Note that weights equal to 0 produce NaNs, so we replace them with 0 again
            config_.weights_per_class = np.nan_to_num(config_.weights_per_class)

        # Now we can start generating our federated dataset
        fed_dataset = FlexDataset()
        if config_.indexes_per_client is not None:
            for client_name, data in cls.__sample_dataset_with_indexes(cdata, config_):
                fed_dataset[client_name] = data
        elif config_.group_by_feature is not None:
            for client_name, data in cls.__group_by_feature(cdata, config_):
                fed_dataset[client_name] = data
        else:  # sample using weights or features
            remaining_data_indices = np.arange(len(cdata))
            for i in range(config_.n_clients):
                (
                    sub_data_indices,
                    sub_features_indices,
                    remaining_data_indices,
                ) = cls.__sample(rng, remaining_data_indices, cdata, config_, i)

                fed_dataset[config_.client_names[i]] = FlexDataObject(
                    X_data=cdata.X_data[sub_data_indices][:, sub_features_indices]
                    if len(cdata.X_data.shape) > 1
                    else cdata.X_data[sub_data_indices],
                    y_data=cdata.y_data[sub_data_indices]
                    if cdata.y_data is not None
                    else None,
                )

        return fed_dataset

    @classmethod
    def __group_by_feature(cls, cdata: FlexDataObject, config: FlexDatasetConfig):
        f_index = config.group_by_feature
        feat_to_cname = {}
        x_data = defaultdict(list)
        y_data = defaultdict(list)
        for i, (x, y) in enumerate(cdata):
            feature = str(x[f_index])  # Use str to make every feature hashable
            if feature not in feat_to_cname:
                feat_to_cname[feature] = i
            x_data[feat_to_cname[feature]].append(x)
            y_data[feat_to_cname[feature]].append(y)
        for k in x_data:
            yield k, FlexDataObject(
                X_data=np.asarray(x_data[k]), y_data=np.asarray(y_data[k])
            )

    @classmethod
    def __sample_dataset_with_indexes(
        cls, data: FlexDataObject, config: FlexDatasetConfig
    ):
        """Iterable function that associates a client with its data, when a list of indexes is given for
        each client.

        Args:
            data (FlexDataObject): Centralizaed dataset represented as a FlexDataObject.
            config (FlexDatasetConfig): Configuration used to federate a FlexDataObject.

        Yields:
            tuple (Tuple): a tuple whose first item is the client name and the second one is the indexes of
            the dataset associated to such client.

        """
        for idx, name in zip(config.indexes_per_client, config.client_names):
            yield name, FlexDataObject(
                X_data=data.X_data[idx],
                y_data=data.y_data[idx] if data.y_data is not None else None,
            )

    @classmethod
    def __sample(
        cls,
        rng: np.random.Generator,
        data_indices: npt.NDArray[np.int_],
        data: FlexDataObject,
        config: FlexDatasetConfig,
        client_i: int,
    ) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Function to sample indices from a FlexDataObject as especified by a FlexDatasetConfig.

        Args:
            rng (np.random.Generator): Random number generator used to sample.
            data_indices (npt.NDArray[np.int_]): Array of available data indices to sample from.
            data (FlexDataObject): Centralizaed dataset represented as a FlexDataObject.
            config (FlexDatasetConfig): Configuration used to federate a FlexDataObject.
            client_i (int): Position of client which will be identified with the generated sample.

        Returns:
            sample_indices (Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]): it returns
            the sampled data indices, the sampled feature indices and the data indices which were not used for
            the sampled data indices. Note that, the latter are only used for the config.replacement option, otherwise
            it contains all the provided data_indices.
        """
        if config.features_per_client is None:
            sub_data_indices, sub_features_indices = cls.__sample_with_weights(
                rng, data_indices, data, config, client_i
            )
        else:
            sub_data_indices, sub_features_indices = cls.__sample_with_features(
                rng, data_indices, data, config, client_i
            )

        # Update remaning data indices
        remaining_data_indices = (
            data_indices
            if config.replacement
            else np.array(list(set(data_indices) - set(sub_data_indices)))
        )

        return sub_data_indices, sub_features_indices, remaining_data_indices

    @classmethod
    def __sample_with_weights(
        cls,
        rng: np.random.Generator,
        data_indices: npt.NDArray[np.int_],
        data: FlexDataObject,
        config: FlexDatasetConfig,
        client_i: int,
    ):
        """Especialized function to sample indices from a FlexDataObject as especified by a FlexDatasetConfig.
            It takes into consideration the config.weights and config.weights_per_class option and applies it.
            If no config.weights and no config.weights_per_class are provided, then we consider that the weights \
            are the same for all the clients.

        Args:
            rng (np.random.Generator): Random number generator used to sample.
            data_indices (npt.NDArray[np.int_]): Array of available data indices to sample from.
            data (FlexDataObject): Centralizaed dataset represented as a FlexDataObject.
            config (FlexDatasetConfig): Configuration used to federate a FlexDataObject.
            client_i (int): Position of client which will be identified with the generated sample.

        Returns:
            sample_indices (Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]): it returns
            the sampled data indices and all the feature indices.
        """
        if config.weights_per_class is not None:
            data_proportion = None
        elif config.weights is not None:
            data_proportion = floor(len(data) * config.weights[client_i])
        else:  # No weights provided
            data_proportion = floor(len(data) / config.n_clients)

        if data_proportion is not None:
            sub_data_indices = rng.choice(data_indices, data_proportion, replace=False)
        else:  # apply weights_per_class
            sub_data_indices = np.array([], dtype="uint32")
            sorted_classes = np.sort(np.unique(data.y_data))
            all_indices = np.arange(len(data))
            for j, c in enumerate(sorted_classes):
                available_class_indices = all_indices[data.y_data == c]
                proportion_per_class = floor(
                    len(available_class_indices) * config.weights_per_class[client_i][j]
                )
                selected_class_indices = rng.choice(
                    available_class_indices, proportion_per_class, replace=False
                )
                sub_data_indices = np.concatenate(
                    (sub_data_indices, selected_class_indices)
                )

        # Sample feature indices
        sub_features_indices = slice(
            None
        )  # Default slice for features, it includes all the features
        return sub_data_indices, sub_features_indices

    @classmethod
    def __configure_weights_per_class(
        cls, rng: np.random.Generator, config: FlexDatasetConfig, data: FlexDataObject
    ):
        sorted_classes = np.sort(np.unique(data.y_data))
        assigned_classes = []
        if isinstance(config.classes_per_client, int):
            for _ in range(config.n_clients):
                n = rng.choice(
                    sorted_classes, size=config.classes_per_client, replace=False
                )
                assigned_classes.append(n)
            config.classes_per_client = assigned_classes
        elif isinstance(config.classes_per_client, tuple):
            num_classes_per_client = rng.integers(
                low=config.classes_per_client[0],
                high=config.classes_per_client[1] + 1,
                size=config.n_clients,
            )
            for c in num_classes_per_client:
                n = rng.choice(sorted_classes, size=c, replace=False)
                assigned_classes.append(n)
            config.classes_per_client = assigned_classes

        config.weights_per_class = np.zeros((config.n_clients, len(sorted_classes)))
        for client_i, clasess_at_client_i in enumerate(config.classes_per_client):
            for class_j, label in enumerate(sorted_classes):
                if label in clasess_at_client_i:
                    if config.weights is None:
                        config.weights_per_class[client_i, class_j] = 1
                    else:
                        config.weights_per_class[client_i, class_j] = config.weights[
                            client_i
                        ] / len(clasess_at_client_i)

    @classmethod
    def __sample_with_features(
        cls,
        rng,
        data_indices: npt.NDArray[np.int_],
        data: FlexDataObject,
        config: FlexDatasetConfig,
        client_i: int,
    ):
        """Especialized function to sample indices from a FlexDataObject as especified by a FlexDatasetConfig.
            It takes into consideration the config.features_per_client option and applies it. Weights are applied
            the same as in __sample_with_weights.

        Args:
            rng (np.random.Generator): Random number generator used to sample.
            data_indices (npt.NDArray[np.int_]): Array of available data indices to sample from.
            data (FlexDataObject): Centralized dataset represented as a FlexDataObject.
            config (FlexDatasetConfig): Configuration used to federate a FlexDataObject.
            client_i (int): Position of client which will be identified with the generated sample.

        Returns:
            sample_indices (Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]): it returns the sampled data indices
            and the sampled feature indices.
        """
        # Sample data indices
        sub_data_indices, _ = cls.__sample_with_weights(
            rng, data_indices, data, config, client_i
        )

        # Sample feature indices
        feature_indices = np.arange(data.X_data.shape[1])
        if isinstance(  # We have a fixed number of features per client
            config.features_per_client, int
        ):
            sub_features_indices = rng.choice(
                feature_indices, config.features_per_client, replace=False
            )
        elif isinstance(  # We have a maximum and a minimum of features per client
            config.features_per_client, tuple
        ):
            sub_features_indices = rng.choice(
                feature_indices,
                rng.integers(
                    config.features_per_client[0], config.features_per_client[1] + 1
                ),
                replace=False,
            )
        else:  # We have an array of features per client, that is, each client has an set of classes
            sub_features_indices = config.features_per_client[client_i]

        return sub_data_indices, sub_features_indices
