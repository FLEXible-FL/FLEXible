import unittest

import pytest

from flex.data import FlexDatasetConfig
import numpy as np

class TestFlexDatasetConfig(unittest.TestCase):
    def test_missing_params(self):
        a = FlexDatasetConfig()
        with pytest.raises(ValueError):
            a.validate()

    def test_nclients(self):
        a = FlexDatasetConfig(n_clients=1)
        with pytest.raises(ValueError):
            a.validate()

    def test_client_names_error(self):
        a = FlexDatasetConfig(client_names=["Pepe"])
        with pytest.raises(ValueError):
            a.validate()

    def test_weights(self):
        a = FlexDatasetConfig(n_clients=2, weights=[0.5])
        with pytest.raises(ValueError):
            a.validate()

    def test_max_weights(self):
        a = FlexDatasetConfig(n_clients=2, weights=[1.2, 0.3])
        with pytest.raises(ValueError):
            a.validate()

    def test_positive_weights(self):
        a = FlexDatasetConfig(n_clients=2, weights=[-1.2, 0.3])
        with pytest.raises(ValueError):
            a.validate()

    def test_mutually_exclusive_options(self):
        a = FlexDatasetConfig(n_clients=2, classes_per_client=2, features_per_client=2)
        with pytest.raises(ValueError):
            a.validate()

    def test_classes_per_client_tuple(self):
        a = FlexDatasetConfig(n_clients=2, classes_per_client=(1, 2, 3))
        with pytest.raises(ValueError):
            a.validate()
        a.classes_per_client = (1,)
        with pytest.raises(ValueError):
            a.validate()

    def test_classes_per_client_arr(self):
        a = FlexDatasetConfig(n_clients=2, classes_per_client=[0, [0, 2], 1])
        with pytest.raises(ValueError):
            a.validate()

    def test_features_per_client_arr(self):
        a = FlexDatasetConfig(n_clients=2, features_per_client=[[0, 1], [2, 3], [4]])
        with pytest.raises(ValueError):
            a.validate()

    def test_features_per_client_tuple(self):
        a = FlexDatasetConfig(n_clients=2, features_per_client=(1, 2, 3))
        with pytest.raises(ValueError):
            a.validate()
        a.features_per_client = (1,)
        with pytest.raises(ValueError):
            a.validate()

    def test_features_per_client_w_replacement(self):
        a = FlexDatasetConfig(replacement=False, n_clients=2, features_per_client=3)
        with pytest.raises(ValueError):
            a.validate()

    def test_incompatible_options_w_indexes_per_client(self):
        a = FlexDatasetConfig(
            n_clients=3, features_per_client=3, indexes_per_client=[[2]]
        )
        b = FlexDatasetConfig(
            n_clients=3, classes_per_client=3, indexes_per_client=[[2]]
        )
        with pytest.raises(ValueError):
            a.validate()
        with pytest.raises(ValueError):
            b.validate()

    def test_indexes_per_client_w_number_of_clients(self):
        a = FlexDatasetConfig(n_clients=2, indexes_per_client=[])
        with pytest.raises(ValueError):
            a.validate()

    def test_indexes_per_client_w_client_names(self):
        a = FlexDatasetConfig(
            client_names=["clarisa", "Randuin"], indexes_per_client=[]
        )
        with pytest.raises(ValueError):
            a.validate()

    def test_weights_incompatibility(self):
        a = FlexDatasetConfig(n_clients=2, weights=[1, 2], weights_per_class=np.ones((2, 2)))
        with pytest.raises(ValueError):
            a.validate()

    def test_weights_per_class_shape(self):
        a = FlexDatasetConfig(n_clients=2, weights_per_class=np.ones((1,)))
        with pytest.raises(ValueError):
            a.validate()

    def test_weights_per_class_size(self):
        a = FlexDatasetConfig(n_clients=2, weights_per_class=np.ones((1, 2)))
        with pytest.raises(ValueError):
            a.validate()