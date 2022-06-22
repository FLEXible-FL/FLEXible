import copy
from typing import List

import numpy as np
from numpy.random import default_rng

from flex.data import FlexDataObject, FlexDataset, FlexDatasetConfig


class FlexDataDistribution(object):
    __create_key = object()

    def __init__(self, create_key: object = None) -> None:
        assert (
            create_key == FlexDataDistribution.__create_key
        ), """FlexDataDistribution objects must be created using FlexDataDistribution.from_state or
        FlexDataDistribution.iid_distribution"""

    @classmethod
    def from_state(cls, cdata: FlexDataObject, config: FlexDatasetConfig):
        """This function prepare the data from a centralized data structure to a federated one.
        It will run diffetent modifications to federate the data.

        Args:
            cdata (FlexDataObject): Centralized dataset represented as a FlexDataObject.
            state (FlexDatasetConfig): FlexDatasetConfig with the configuration to federate the dataset.

        Returns:
            federated_dataset (FlexDataset): The dataset federated.
        """
        cdata.validate()
        config.validate(cdata)

        config_ = copy.copy(config)  # copy, because we might modify some components
        if not config.replacement and sum(config.weights) > 1:  # normalize weights
            config_.weights = [w / sum(config.weights) for w in config.weights]

        fed_dataset = FlexDataset()
        remaining_data_indices = np.arange(len(cdata))
        for i in range(config.n_clients):
            (
                sub_data_indices,
                sub_features_indices,
                remaining_data_indices,
            ) = cls.__sample(remaining_data_indices, cdata, config_, i)
            sub_dataset = FlexDataObject()
            sub_dataset.X_data = cdata.X_data[
                sub_data_indices, sub_features_indices, ...
            ]
            sub_dataset.y_data = cdata.y_data[sub_data_indices]
            sub_dataset.X_names = cdata.X_names[sub_features_indices]
            sub_dataset.y_names = cdata.y_names
            fed_dataset[i] = sub_dataset

        return fed_dataset

    @classmethod
    def iid_distribution(cls, cdata: FlexDataObject, n_clients: int = 2):
        """Function to create a FlexDataset for an IID experiment.

        Args:
            cdata (FlexDataObject): Centralizaed dataset represented as a FlexDataObject.
            n_clients (int): Number of clients in the Federated Learning experiment. Default 2.

        Returns:
            federated_dataset (FederatedFlexDatasetObject): Federated Dataset,
        """
        # TODO: Once FlexState is finished, and other functions to create the FlexDataset
        # are finished too, continue with this class.
        # return FlexDataset()

    @classmethod
    def __sample(
        cls,
        data_indices: List[int],
        data: FlexDataObject,
        config: FlexDatasetConfig,
        client_i: int,
    ):
        y_classes = np.unique(data.y_data)
        feature_indices = np.arange(data.y_data.shape[1])
        rng = default_rng()
        data_proportion = int(len(data) * config.weights[client_i])
        if (  # Only weights affect to the sampling procedure
            config.classes_per_client is None and config.features_per_client is None
        ):
            sub_features_indices = slice(None)
            sub_data_indices = rng.choice(data_indices, data_proportion, replace=False)
        elif config.classes_per_client is not None:
            if isinstance(  # We have a fixed number of classes per client
                config.classes_per_client, int
            ):
                sub_y_classes = rng.choice(
                    y_classes, config.classes_per_client, replace=False
                )
            elif isinstance(  # We have a maximum and a minimum of classes per client
                config.classes_per_client, tuple
            ):
                sub_y_classes = rng.choice(
                    y_classes,
                    np.random.randint(*config.classes_per_client),
                    replace=False,
                )
            else:  # We have classes assigned for each client
                sub_y_classes = config.classes_per_client[client_i]
            sub_data_indices = rng.choice(
                data_indices[np.isin(data.y_data, sub_y_classes)],
                data_proportion,
                replace=False,
            )
            sub_features_indices = slice(None)
        else:
            if isinstance(
                config.features_per_client, int
            ):  # We have a fixed number of features per client
                sub_features_indices = rng.choice(
                    feature_indices, config.features_per_client, replace=False
                )
            elif isinstance(
                config.features_per_client, tuple
            ):  # We have a maximum and a minimum of classes per client
                sub_features_indices = rng.choice(
                    feature_indices,
                    np.random.randint(*config.features_per_client),
                    replace=False,
                )
            else:
                sub_features_indices = config.features_per_client[client_i]
            sub_data_indices = rng.choice(data_indices, data_proportion, replace=False)
        remaining_data_indices = data_indices
        if not config.replacement:
            remaining_data_indices = list(set(data_indices) - set(sub_data_indices))
        return sub_data_indices, sub_features_indices, remaining_data_indices
