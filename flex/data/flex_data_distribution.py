import copy
from collections import defaultdict
from math import floor
from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt
from numpy.random import default_rng

from flex.data import FlexDataObject, FlexDataset, FlexDatasetConfig


class FlexDataDistribution(object):
    __create_key = object()

    def __init__(self, create_key: object = None) -> None:
        assert (
            create_key == FlexDataDistribution.__create_key
        ), """FlexDataDistribution objects must be created using FlexDataDistribution.from_config or
        FlexDataDistribution.iid_distribution"""

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
        d = defaultdict()
        for idx, (x, y) in enumerate(cdata):
            client_name = clustering_func(x, y)
            if d.get(client_name) is None:
                d[client_name] = [idx]
            else:
                d[client_name].append(idx)
        config = FlexDatasetConfig(
            client_names=list(d.keys()), indexes_per_client=list(d.values())
        )
        return cls.from_config(cdata, config)

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

        if (  # normalize weights if no replacement
            config.weights is not None
            and not config.replacement
            and sum(config.weights) > 1
            and config.classes_per_client is None
        ):
            config_.weights = np.array(
                [w / sum(config.weights) for w in config.weights]
            )

        if config_.client_names is not None and config_.n_clients is not None:
            common_min = min(config_.n_clients, len(config_.client_names))
            config_.n_clients = common_min
            config_.client_names = config_.client_names[:common_min]
        elif config_.client_names is not None:
            config_.n_clients = len(config_.client_names)
        elif config_.n_clients is not None:
            config_.client_names = list(range(config_.n_clients))

        fed_dataset = FlexDataset()
        if config_.indexes_per_client is not None:
            for client_name, data in cls.__sample_dataset_with_indexes(cdata, config_):
                fed_dataset[client_name] = data
        else:
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
        if config.classes_per_client is None and config.features_per_client is None:
            sub_data_indices, sub_features_indices = cls.__sample_only_with_weights(
                rng, data_indices, data, config, client_i
            )
        elif config.classes_per_client is not None:
            sub_data_indices, sub_features_indices = cls.__sample_with_classes(
                rng, data_indices, data, config, client_i
            )
        else:  # elif config.features_per_client is not None
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
    def __sample_only_with_weights(
        cls,
        rng: np.random.Generator,
        data_indices: npt.NDArray[np.int_],
        data: FlexDataObject,
        config: FlexDatasetConfig,
        client_i: int,
    ):
        """Especialized function to sample indices from a FlexDataObject as especified by a FlexDatasetConfig.
            It takes into consideration the config.weights option and applies it. If no config.weights are
            provided, then we consider that the weights are the same for all the clients.

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
        # Sample data indices
        if config.weights is None:  # No weights provided
            data_proportion = floor(len(data) / config.n_clients)
        else:
            data_proportion = floor(len(data) * config.weights[client_i])

        sub_data_indices = rng.choice(data_indices, data_proportion, replace=False)
        # Sample feature indices
        sub_features_indices = slice(
            None
        )  # Default slice for features, it includes all the features
        return sub_data_indices, sub_features_indices

    @classmethod
    def __sample_with_classes(
        cls,
        rng: np.random.Generator,
        data_indices: npt.NDArray[np.int_],
        data: FlexDataObject,
        config: FlexDatasetConfig,
        client_i: int,
    ):
        """Especialized function to sample indices from a FlexDataObject as especified by a FlexDatasetConfig.
            It takes into consideration the config.classes_per_client option and applies it. If config.weights are
            provided, then we consider the current weight is shared equally among the sampled classes. Otherwise,
            we sample all the elements with the sampled classes.

        Args:
            rng (np.random.Generator): Random number generator used to sample.
            data_indices (npt.NDArray[np.int_]): Array of available data indices to sample from.
            data (FlexDataObject): Centralizaed dataset represented as a FlexDataObject.
            config (FlexDatasetConfig): Configuration used to federate a FlexDataObject.
            client_i (int): Position of client which will be identified with the generated sample.

        Returns:
            sample_indices (Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]): it returns the sampled data indices
            and all the feature indices.
        """
        # Sample data indices
        y_data_available = data.y_data[data_indices]
        y_classes_available = np.unique(y_data_available)

        if isinstance(  # We have a fixed number of classes per client
            config.classes_per_client, int
        ):
            sub_y_classes = rng.choice(
                y_classes_available, config.classes_per_client, replace=False
            )
        elif isinstance(  # We have a maximum and a minimum of classes per client
            config.classes_per_client, tuple
        ):
            # if config == FlexDatasetConfig(seed=1, n_clients=2, classes_per_client=(2, 3), weights=[0.25, 1]):
            #     breakpoint()
            sub_y_classes = rng.choice(
                y_classes_available,
                rng.integers(
                    config.classes_per_client[0], config.classes_per_client[1] + 1
                ),
                replace=False,
            )
        else:  # We have classes assigned for each client
            sub_y_classes = np.array(config.classes_per_client[client_i])

        sub_data_indices = data_indices[np.isin(y_data_available, sub_y_classes)]
        if config.weights is not None:
            len_all_data_available = len(sub_data_indices)
            sub_data_indices = np.array([], dtype="uint32")
            if len(sub_y_classes.shape) > 0:
                for (
                    c
                ) in (
                    sub_y_classes
                ):  # Ensure that each class is represented in the sample
                    available_data_indices = data_indices[y_data_available == c]
                    data_proportion = floor(
                        len_all_data_available
                        * config.weights[client_i]
                        / len(sub_y_classes)
                    )
                    tmp_indices = rng.choice(
                        available_data_indices, data_proportion, replace=False
                    )
                    sub_data_indices = np.concatenate((sub_data_indices, tmp_indices))
            else:
                sub_data_indices = data_indices[
                    np.isin(y_data_available, sub_y_classes)
                ]
                data_proportion = floor(
                    len(sub_data_indices) * config.weights[client_i]
                )
                sub_data_indices = rng.choice(
                    sub_data_indices, data_proportion, replace=False
                )

        # Sample feature indices
        sub_features_indices = slice(
            None
        )  # Default slice for features, it includes all the features

        return sub_data_indices, sub_features_indices

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
            the same as in __sample_only_with_weights.

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
        sub_data_indices, _ = cls.__sample_only_with_weights(
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
