import copy
from math import floor

import numpy as np
import numpy.typing as npt
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
        rng = default_rng(seed=config.seed)

        config_ = copy.copy(config)  # copy, because we might modify some components

        if (  # normalize weights
            config.weights is not None
            and not config.replacement
            and sum(config.weights) > 1
        ):
            config_.weights = [w / sum(config.weights) for w in config.weights]
        # elif config.weights is None and config.classes_per_client is None:
        #    config_.weights = [1 / config.n_clients for _ in range(config.n_clients)]

        fed_dataset = FlexDataset()
        remaining_data_indices = np.arange(len(cdata))
        for i in range(config_.n_clients):
            (
                sub_data_indices,
                sub_features_indices,
                remaining_data_indices,
            ) = cls.__sample(rng, remaining_data_indices, cdata, config_, i)

            fed_dataset[i] = FlexDataObject(
                X_data=cdata.X_data[sub_data_indices][:, sub_features_indices],
                y_data=cdata.y_data[sub_data_indices]
                if cdata.y_data is not None
                else None,
                X_names=cdata.X_names[sub_features_indices]
                if cdata.X_names is not None
                else None,
                y_names=cdata.y_names,
            )

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
        rng,
        data_indices: npt.NDArray[np.int_],
        data: FlexDataObject,
        config: FlexDatasetConfig,
        client_i: int,
    ):
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
        rng,
        data_indices: npt.NDArray[np.int_],
        data: FlexDataObject,
        config: FlexDatasetConfig,
        client_i: int,
    ):
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
        rng,
        data_indices: npt.NDArray[np.int_],
        data: FlexDataObject,
        config: FlexDatasetConfig,
        client_i: int,
    ):
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
            sub_y_classes = config.classes_per_client[client_i]

        sub_data_indices = data_indices[np.isin(y_data_available, sub_y_classes)]
        if config.weights is not None:
            len_all_data_available = len(sub_data_indices)
            sub_data_indices = np.array([], dtype="int64")
            if not isinstance(sub_y_classes, int):
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
