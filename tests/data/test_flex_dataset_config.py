import unittest

import numpy as np
import pytest

from flex.data import FlexDataObject, FlexDatasetConfig


class TestFlexDatasetConfig(unittest.TestCase):
    def test_nclients(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        a = FlexDatasetConfig(n_clients=1)
        with pytest.raises(ValueError):
            a.validate(fcd)

    def test_weights(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        a = FlexDatasetConfig(n_clients=2, weights=[0.5])
        with pytest.raises(ValueError):
            a.validate(fcd)

    def test_max_weights(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        a = FlexDatasetConfig(n_clients=2, weights=[1.2, 0.3])
        with pytest.raises(ValueError):
            a.validate(fcd)

    def test_mutually_exclusive_options(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        a = FlexDatasetConfig(classes_per_client=2, features_per_client=2)
        with pytest.raises(ValueError):
            a.validate(fcd)

    def test_replacement_and_features_per_client(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        a = FlexDatasetConfig(replacement=False, features_per_client=2)
        with pytest.raises(ValueError):
            a.validate(fcd)

    def test_classes_per_client_int(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        a = FlexDatasetConfig(classes_per_client=20)
        with pytest.raises(ValueError):
            a.validate(fcd)
        a.classes_per_client = 0
        with pytest.raises(ValueError):
            a.validate(fcd)

    def test_classes_per_client_tuple(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        a = FlexDatasetConfig(classes_per_client=(1, 2, 3))
        with pytest.raises(ValueError):
            a.validate(fcd)
        a.classes_per_client = (1,)
        with pytest.raises(ValueError):
            a.validate(fcd)

    def test_classes_per_client_tuple_bis(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        a = FlexDatasetConfig(classes_per_client=(3, 2))
        with pytest.raises(ValueError):
            a.validate(fcd)

    def test_classes_per_client_arr(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        a = FlexDatasetConfig(classes_per_client=[y_data[0], y_data[1], y_data])
        with pytest.raises(ValueError):
            a.validate(fcd)

    def test_features_per_client_int(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        a = FlexDatasetConfig(features_per_client=20)
        with pytest.raises(ValueError):
            a.validate(fcd)
        a.features_per_client = 0
        with pytest.raises(ValueError):
            a.validate(fcd)

    def test_features_per_client_tuple(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        a = FlexDatasetConfig(features_per_client=(1, 2, 3))
        with pytest.raises(ValueError):
            a.validate(fcd)
        a.features_per_client = (1,)
        with pytest.raises(ValueError):
            a.validate(fcd)

    def test_features_per_client_tuple_bis(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        a = FlexDatasetConfig(features_per_client=(3, 2))
        with pytest.raises(ValueError):
            a.validate(fcd)

    def test_features_per_client_arr(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        a = FlexDatasetConfig(features_per_client=[[0, 1], [2, 3], [4]])
        with pytest.raises(ValueError):
            a.validate(fcd)
