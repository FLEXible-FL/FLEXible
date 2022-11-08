import unittest

import numpy as np
import pytest

from flex.data import FlexDatasetConfig
from flex.data.flex_dataset_config import InvalidConfig


class TestFlexDatasetConfig(unittest.TestCase):
    def test_nclients(self):
        a = FlexDatasetConfig(n_clients=1)
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_client_names_error(self):
        a = FlexDatasetConfig(client_names=["Pepe"])
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_weights(self):
        a = FlexDatasetConfig(weights=[0.5])
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_max_weights(self):
        a = FlexDatasetConfig(weights=[1.2, 0.3])
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_positive_weights(self):
        a = FlexDatasetConfig(weights=[-1.2, 0.3])
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_mutually_exclusive_options(self):
        a = FlexDatasetConfig(classes_per_client=2, features_per_client=2)
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_classes_per_client_tuple(self):
        a = FlexDatasetConfig(classes_per_client=(1, 2, 3))
        with pytest.raises(InvalidConfig):
            a.validate()
        a.classes_per_client = (1,)
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_classes_per_client_arr(self):
        a = FlexDatasetConfig(classes_per_client=[0, [0, 2], 1])
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_features_per_client_arr(self):
        a = FlexDatasetConfig(
            features_per_client=[[0, 1], [2, 3], [4]], replacement=True
        )
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_features_per_client_tuple(self):
        a = FlexDatasetConfig(features_per_client=(1, 2, 3), replacement=True)
        with pytest.raises(InvalidConfig):
            a.validate()
        a.features_per_client = (1,)
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_features_per_client_w_replacement(self):
        a = FlexDatasetConfig(replacement=False, features_per_client=3)
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_incompatible_options_w_indexes_per_client(self):
        a = FlexDatasetConfig(
            n_clients=3, features_per_client=3, indexes_per_client=[[2]]
        )
        b = FlexDatasetConfig(
            n_clients=3, classes_per_client=3, indexes_per_client=[[2]]
        )
        with pytest.raises(InvalidConfig):
            a.validate()
        with pytest.raises(InvalidConfig):
            b.validate()

    def test_indexes_per_client_w_number_of_clients(self):
        a = FlexDatasetConfig(replacement=False, indexes_per_client=[])
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_indexes_per_client_w_replacement(self):
        a = FlexDatasetConfig(replacement=True, indexes_per_client=[])
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_weights_incompatibility(self):
        a = FlexDatasetConfig(weights=[1, 2], weights_per_class=np.ones((2, 2)))
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_weights_per_class_incompatibility(self):
        a = FlexDatasetConfig(classes_per_client=2, weights_per_class=np.ones((2, 2)))
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_weights_per_class_shape(self):
        a = FlexDatasetConfig(weights_per_class=np.ones((1,)))
        with pytest.raises(InvalidConfig):
            a.validate()

    def test_weights_per_class_size(self):
        a = FlexDatasetConfig(n_clients=3, weights_per_class=np.ones((1, 2)))
        with pytest.raises(InvalidConfig):
            a.validate()
