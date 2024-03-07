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
import copy
from collections import defaultdict
from math import floor
from typing import Callable, Tuple

import numpy as np
import numpy.typing as npt
from numpy.random import default_rng

from flex.data import Dataset, FedDataset, FedDatasetConfig
from flex.data.lazy_indexable import LazyIndexable


class FedDataDistribution(object):
    __create_key = object()

    def __init__(self, create_key: object = None) -> None:
        assert (
            create_key == FedDataDistribution.__create_key
        ), """FedDataDistribution objects must be created using FedDataDistribution.from_config or
        FedDataDistribution.iid_distribution"""

    @classmethod
    def from_config_with_torchtext_dataset(cls, data, config: FedDatasetConfig):
        """This function federates a centralized torchtext dataset given a FlexDatasetConfig.
        This function will transform the torchtext dataset into a Dataset and then it will
        federate it.

        Args:
        -----
            data (Dataset): The torchtext dataset
            config (FedDatasetConfig): FlexDatasetConfig with the configuration to federate the centralized dataset.
        """
        centralized_data = Dataset.from_torchtext_dataset(data)
        return cls.from_config(centralized_data, config)

    @classmethod
    def from_config_with_tfds_image_dataset(cls, data, config: FedDatasetConfig):
        """This function federates a centralized tensorflow dataset given a FlexDatasetConfig.
        This function will transform a dataset from the tensorflow_datasets module into a Dataset
        and then it will federate it.

        Args:
        -----
            data (Dataset): The tensorflow dataset
            config (FedDatasetConfig): FlexDatasetConfig with the configuration to federate the centralized dataset.
        """
        centralized_data = Dataset.from_tfds_image_dataset(data)
        return cls.from_config(centralized_data, config)

    @classmethod
    def from_config_with_tfds_text_dataset(
        cls, data, config: FedDatasetConfig, X_columns: list, label_columns: list
    ):
        """This function federates a centralized tensorflow dataset given a FlexDatasetConfig.
        This function will transform a dataset from the tensorflow_datasets module into a Dataset
        and then it will federate it.

        Args:
        -----
            data (Dataset): The tensorflow dataset
            config (FedDatasetConfig): FlexDatasetConfig with the configuration to federate the centralized dataset.
            X_columns (List): List that contains the columns names for the input features.
            label_columns (List): List that contains the columns names for the output features.
        """
        centralized_data = Dataset.from_tfds_text_dataset(
            data, X_columns, label_columns
        )
        return cls.from_config(centralized_data, config)

    @classmethod
    def from_config_with_torchvision_dataset(cls, data, config: FedDatasetConfig):
        """This function federates a centralized torchvision dataset given a FlexDatasetConfig.
        This function will transform a dataset from the torchvision module into a Dataset
        and then it will federate it.

        Args:
        -----
            data (Dataset): The torchvision dataset
            config (FedDatasetConfig): FlexDatasetConfig with the configuration to federate the centralized dataset.
        """
        centralized_data = Dataset.from_torchvision_dataset(data)
        return cls.from_config(centralized_data, config)

    @classmethod
    def from_config_with_huggingface_dataset(
        cls,
        data,
        config: FedDatasetConfig,
        X_columns: list,
        label_columns: list = None,
    ):
        """This function federates a centralized hugginface dataset given a FlexDatasetConfig.
        This function will transform a dataset from the HuggingFace Hub datasets into a Dataset
        and then it will federate it.

        Args:
        -----
            data (Union[datasets.arrow_dataset.Dataset, str]): The hugginface dataset to federate.
            config (FedDatasetConfig): FlexDatasetConfig with the configuration to federate the centralized dataset.
            X_coluns (List[str]): List with the names of the columns to load.
            label_columns (list): List with the names of the label columns.
        """
        centralized_data = Dataset.from_huggingface_dataset(
            data, X_columns, label_columns
        )
        return cls.from_config(centralized_data, config)

    @classmethod
    def from_clustering_func(cls, centralized_data: Dataset, clustering_func: Callable):
        """This function federates data into nodes by means of a clustering function, that outputs
        to which node (cluster) a data point belongs.

        Args:
        -----
            centralized_data (Dataset): Centralized dataset represented as a FlexDataObject.
            clustering_func (Callable): function that receives as arguments a pair of x and y elements from centralized_data
            and returns the name of the node (cluster) that should own it, the returned type must be Hashable.
            Note that we only support one node (cluster) per data point.

        Returns:
        --------
            federated_dataset (FedDataset): The federated dataset.
        """
        d = defaultdict(list)
        for idx, (x, y) in enumerate(centralized_data):
            node_name = clustering_func(x, y)
            d[node_name].append(idx)

        config = FedDatasetConfig(
            n_nodes=len(d),
            node_ids=list(d.keys()),
            indexes_per_node=list(d.values()),
            replacement=False,
        )
        return cls.from_config(centralized_data, config)

    @classmethod
    def iid_distribution(cls, centralized_data: Dataset, n_nodes: int = 2):
        """Function to create a FedDataset for an IID experiment. We consider the simplest situation
        in which the data is distributed by giving the same amount of data to each node.

        Args:
        -----
            centralized_data (Dataset): Centralized dataset represented as a FlexDataObject.
            n_nodes (int): Number of nodes in the Federated Learning experiment. Default 2.

        Returns:
        --------
            federated_dataset (FedDataset): The federated dataset.
        """
        config = FedDatasetConfig(n_nodes=n_nodes)
        return FedDataDistribution.from_config(centralized_data, config)

    @classmethod
    def from_config(cls, centralized_data: Dataset, config: FedDatasetConfig):
        """This function prepare the data from a centralized data structure to a federated one.
        It will run different modifications to federate the data.

        Args:
        -----
            centralized_data (Dataset): Centralized dataset represented as a FlexDataObject.
            config (FedDatasetConfig): FlexDatasetConfig with the configuration to federate the centralized dataset.

        Returns:
        --------
            federated_dataset (FedDataset): The federated dataset.
        """
        centralized_data.validate()
        config.validate()
        rng = default_rng(seed=config.seed)

        config_ = copy.deepcopy(config)  # copy, because we might modify some components

        if config.node_ids is None:
            config_.node_ids = list(range(config_.n_nodes))

        if config.keep_labels is None:
            config_.keep_labels = [True] * config_.n_nodes

        labels = None
        if centralized_data.y_data is not None:
            labels = centralized_data.y_data.to_numpy()

        # Normalize weights when no replacement
        if (
            not config_.replacement
            and config_.weights is not None
            and sum(config_.weights) > 1
        ):
            config_.weights = np.array(
                [w / sum(config.weights) for w in config.weights]
            )
        # Ensure that labels_per_node is translated to weights_per_label
        if config_.labels_per_node is not None:
            cls.__configure_weights_per_class(rng, config_, labels)
        # Normalize weights_per_label when no replacement
        if (
            not config_.replacement
            and config_.weights_per_label is not None
            and any(np.sum(config_.weights_per_label, axis=0) > 1)
        ):
            with np.errstate(divide="ignore", invalid="ignore"):  # raise no warnings
                config_.weights_per_label = config_.weights_per_label / np.sum(
                    config_.weights_per_label, axis=0
                )
            # Note that weights equal to 0 produce NaNs, so we replace them with 0 again
            config_.weights_per_label = np.nan_to_num(config_.weights_per_label)

        # Now we can start generating our federated dataset
        fed_dataset = FedDataset()
        if config_.indexes_per_node is not None:
            for node_name, data in cls.__sample_dataset_with_indexes(
                centralized_data, config_
            ):
                fed_dataset[node_name] = data
        elif config_.group_by_label_index is not None:
            for node_name, data in cls.__group_by_label_index(
                centralized_data, config_
            ):
                fed_dataset[node_name] = data
        else:  # sample using weights or features
            remaining_data_indices = np.arange(len(labels))
            for i in range(config_.n_nodes):
                if config_.shuffle:
                    rng.shuffle(remaining_data_indices)
                keep_y_data = (
                    centralized_data.y_data is not None and config_.keep_labels[i]
                )
                (
                    sub_data_indices,
                    sub_features_indices,
                    remaining_data_indices,
                ) = cls.__sample(
                    rng, remaining_data_indices, centralized_data, labels, config_, i
                )
                X_data = centralized_data.X_data[sub_data_indices]
                if config.features_per_node is not None:
                    X_data = X_data.to_numpy()
                    X_data = LazyIndexable(X_data[:, sub_features_indices], len(X_data))
                if keep_y_data:
                    y_data = centralized_data.y_data[sub_data_indices]
                else:
                    y_data = None
                fed_dataset[config_.node_ids[i]] = Dataset(X_data=X_data, y_data=y_data)

        return fed_dataset

    @classmethod
    def __group_by_label_index(
        cls, centralized_data: Dataset, config: FedDatasetConfig
    ):
        label_index = config.group_by_label_index
        label_to_node_id = {}
        y_data = defaultdict(list)
        x_data_indexes = defaultdict(list)
        for i, y in enumerate(centralized_data.y_data):
            y = list(y)  # TODO: enforce that y is only a list or a tuple
            str_label = str(y.pop(label_index))  # Use str to make every label hashable
            if str_label not in label_to_node_id:
                label_to_node_id[
                    str_label
                ] = i  # Name each node using the first index where the label appears
            x_data_indexes[label_to_node_id[str_label]].append(i)
            if len(y) == 1:
                y = y[0]
            y_data[label_to_node_id[str_label]].append(y)
        for node_id in y_data:
            yield node_id, Dataset(
                X_data=centralized_data.X_data[x_data_indexes[node_id]],
                y_data=LazyIndexable(y_data[node_id], len(y_data[node_id])),
            )

    @classmethod
    def __sample_dataset_with_indexes(cls, data: Dataset, config: FedDatasetConfig):
        """Iterable function that associates a node with its data, when a list of indexes is given for
        each node.

        Args:
        -----
            data (Dataset): Centralizaed dataset represented as a FlexDataObject.
            config (FedDatasetConfig): Configuration used to federate a FlexDataObject.

        Yields:
        -------
            tuple (Tuple): a tuple whose first item is the node name and the second one is the indexes of
            the dataset associated to such node.

        """
        for idx, name, keep in zip(
            config.indexes_per_node, config.node_ids, config.keep_labels
        ):
            yield name, Dataset(
                X_data=data.X_data[idx],
                y_data=data.y_data[idx] if data.y_data is not None and keep else None,
            )

    @classmethod
    def __sample(
        cls,
        rng: np.random.Generator,
        data_indices: npt.NDArray[np.int_],
        data: Dataset,
        labels: npt.ArrayLike,
        config: FedDatasetConfig,
        node_i: int,
    ) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        """Function to sample indices from a FlexDataObject as especified by a FlexDatasetConfig.

        Args:
        -----
            rng (np.random.Generator): Random number generator used to sample.
            data_indices (npt.NDArray[np.int_]): Array of available data indices to sample from.
            data (Dataset): Centralizaed dataset represented as a FlexDataObject.
            config (FedDatasetConfig): Configuration used to federate a FlexDataObject.
            node_i (int): Position of node which will be identified with the generated sample.

        Returns:
        --------
            sample_indices (Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]): it returns
            the sampled data indices, the sampled feature indices and the data indices which were not used for
            the sampled data indices. Note that, the latter are only used for the config.replacement option, otherwise
            it contains all the provided data_indices.
        """
        # Sample feature indices
        sub_features_indices = cls.__sample_features(rng, data, config, node_i)
        # Sample data indices
        sub_data_indices = cls.__sample_with_weights(
            rng, data_indices, labels, config, node_i
        )
        # Update remaining data indices
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
        labels: npt.ArrayLike,
        config: FedDatasetConfig,
        node_i: int,
    ):
        """Especialized function to sample indices from a FlexDataObject as especified by a FlexDatasetConfig.
            It takes into consideration the config.weights and config.weights_per_label option and applies it.
            If no config.weights and no config.weights_per_label are provided, then we consider that the weights \
            are the same for all the nodes.

        Args:
        -----
            rng (np.random.Generator): Random number generator used to sample.
            data_indices (npt.NDArray[np.int_]): Array of available data indices to sample from.
            data (Dataset): Centralizaed dataset represented as a FlexDataObject.
            config (FedDatasetConfig): Configuration used to federate a FlexDataObject.
            node_i (int): Position of node which will be identified with the generated sample.

        Returns:
        --------
            sample_indices (Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]): it returns
            the sampled data indices and all the feature indices.
        """
        if config.weights_per_label is not None:
            data_proportion = None
        elif config.weights is not None:
            data_proportion = floor(len(labels) * config.weights[node_i])
        else:  # No weights provided
            data_proportion = floor(len(labels) / config.n_nodes)

        if data_proportion is not None:
            sub_data_indices = data_indices[:data_proportion]
        else:  # apply weights_per_label
            sub_data_indices = np.array([], dtype="uint32")
            sorted_labels = np.sort(np.unique(labels))
            available_indices = copy.deepcopy(data_indices)
            if config.shuffle:
                rng.shuffle(available_indices)
            proportion_per_label = {}
            for j, label in enumerate(sorted_labels):
                available_class_indices = sum(labels == label)
                proportion_per_label[label] = floor(
                    available_class_indices * config.weights_per_label[node_i][j]
                )
            for label in proportion_per_label:
                available_class_indices = available_indices[
                    labels[available_indices] == label
                ]
                selected_class_indices = available_class_indices[
                    : proportion_per_label[label]
                ]
                sub_data_indices = np.concatenate(
                    (sub_data_indices, selected_class_indices)
                )
                available_indices = np.array(
                    list(set(available_indices) - set(selected_class_indices))
                )

        return sub_data_indices

    @classmethod
    def __configure_weights_per_class(
        cls, rng: np.random.Generator, config: FedDatasetConfig, labels: npt.ArrayLike
    ):
        sorted_labels = np.sort(np.unique(labels))
        assigned_labels = []
        if isinstance(config.labels_per_node, int):
            histogram = np.zeros_like(sorted_labels)
            for _ in range(config.n_nodes):
                individual_assigned_labels = []
                for _ in range(config.labels_per_node):
                    most_frequent = np.max(histogram)
                    available_labels_indexes = np.arange(len(sorted_labels))
                    tmp_available_indexes = histogram < most_frequent
                    if sum(tmp_available_indexes) != 0:
                        available_labels_indexes = available_labels_indexes[
                            tmp_available_indexes
                        ]
                    indx = rng.choice(available_labels_indexes, size=1, replace=False)
                    histogram[indx] = histogram[indx] + 1
                    individual_assigned_labels.append(sorted_labels[indx])
                assigned_labels.append(individual_assigned_labels)
            config.labels_per_node = assigned_labels
        elif isinstance(config.labels_per_node, tuple):
            num_labels_per_node = rng.integers(
                low=config.labels_per_node[0],
                high=config.labels_per_node[1] + 1,
                size=config.n_nodes,
            )
            for c in num_labels_per_node:
                n = rng.choice(sorted_labels, size=c, replace=False)
                assigned_labels.append(n)
            config.labels_per_node = assigned_labels

        config.weights_per_label = np.zeros((config.n_nodes, len(sorted_labels)))
        for node_i, clasess_at_node_i in enumerate(config.labels_per_node):
            for class_j, label in enumerate(sorted_labels):
                if label in clasess_at_node_i:
                    if config.weights is None:
                        config.weights_per_label[node_i, class_j] = 1
                    else:
                        config.weights_per_label[node_i, class_j] = config.weights[
                            node_i
                        ] / len(clasess_at_node_i)

    @classmethod
    def __sample_features(
        cls,
        rng: np.random.Generator,
        data: Dataset,
        config: FedDatasetConfig,
        node_i: int,
    ):
        """Especialized function to sample indices from a FlexDataObject as especified by a FlexDatasetConfig.
            It takes into consideration the config.features_per_node option and applies it.

        Args:
        -----
            rng (np.random.Generator): Random number generator used to sample.
            data (Dataset): Centralized dataset represented as a FlexDataObject.
            config (FedDatasetConfig): Configuration used to federate a FlexDataObject.
            node_i (int): Position of node which will be identified with the generated sample.

        Returns:
        --------
            sample_indices (Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]): it returns the sampled data indices
            and the sampled feature indices.
        """

        if config.features_per_node is None:
            sub_features_indices = slice(None)
        else:
            feature_indices = np.arange(len(data.X_data[0]))
            if isinstance(  # We have a fixed number of features per node
                config.features_per_node, int
            ):
                sub_features_indices = rng.choice(
                    feature_indices, config.features_per_node, replace=False
                )
            elif isinstance(  # We have a maximum and a minimum of features per node
                config.features_per_node, tuple
            ):
                sub_features_indices = rng.choice(
                    feature_indices,
                    rng.integers(
                        config.features_per_node[0], config.features_per_node[1] + 1
                    ),
                    replace=False,
                )
            else:  # We have an array of features per node, that is, each node has an set of labels
                sub_features_indices = config.features_per_node[node_i]

        return sub_features_indices
