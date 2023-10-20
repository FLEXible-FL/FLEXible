import unittest

import numpy as np
import pytest

from flex.data import FedDatasetConfig
from flex.data.fed_dataset_config import InvalidConfig


class TestFlexDatasetConfig(unittest.TestCase):
    def test_nclients(self):
        a = FedDatasetConfig(n_nodes=1)
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_client_names_error(self):
        a = FedDatasetConfig(node_ids=["Pepe"])
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_weights(self):
        a = FedDatasetConfig(weights=[0.5])
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_max_weights(self):
        a = FedDatasetConfig(weights=[1.2, 0.3])
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_positive_weights(self):
        a = FedDatasetConfig(weights=[-1.2, 0.3])
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_mutually_exclusive_options(self):
        a = FedDatasetConfig(labels_per_node=2, features_per_node=2)
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_labels_per_client_tuple(self):
        a = FedDatasetConfig(labels_per_node=(1, 2, 3))
        with pytest.raises(InvalidConfig):
            a.validate()
        a.labels_per_node = (1,)
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_labels_per_client_arr(self):
        a = FedDatasetConfig(labels_per_node=[0, [0, 2], 1])
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_features_per_client_arr(self):
        a = FedDatasetConfig(features_per_node=[[0, 1], [2, 3], [4]], replacement=True)
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_features_per_client_tuple(self):
        a = FedDatasetConfig(features_per_node=(1, 2, 3), replacement=True)
        with pytest.raises(InvalidConfig):
            a.validate()
        a.features_per_node = (1,)
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_features_per_client_w_replacement(self):
        a = FedDatasetConfig(replacement=False, features_per_node=3)
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_incompatible_options_w_indexes_per_client(self):
        a = FedDatasetConfig(n_nodes=3, features_per_node=3, indexes_per_node=[[2]])
        b = FedDatasetConfig(n_nodes=3, labels_per_node=3, indexes_per_node=[[2]])
        FedDatasetConfig(n_nodes=3, keep_labels=[True] * 3, indexes_per_node=[[2]])
        with pytest.raises(InvalidConfig):
            a.validate()
        with pytest.raises(InvalidConfig):
            b.validate()

    def test_indexes_per_client_w_number_of_clients(self):
        a = FedDatasetConfig(replacement=False, indexes_per_node=[])
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_indexes_per_client_w_replacement(self):
        a = FedDatasetConfig(replacement=True, indexes_per_node=[])
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_weights_incompatibility(self):
        a = FedDatasetConfig(weights=[1, 2], weights_per_label=np.ones((2, 2)))
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_weights_per_class_incompatibility(self):
        a = FedDatasetConfig(labels_per_node=2, weights_per_label=np.ones((2, 2)))
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_weights_per_class_shape(self):
        a = FedDatasetConfig(weights_per_label=np.ones((1,)))
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_weights_per_class_size(self):
        a = FedDatasetConfig(n_nodes=3, weights_per_label=np.ones((1, 2)))
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_keep_labels(self):
        a = FedDatasetConfig(n_nodes=3, keep_labels=[True])
        with pytest.raises(InvalidConfig):
            a.validate()
