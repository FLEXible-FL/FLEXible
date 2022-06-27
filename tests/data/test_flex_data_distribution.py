import unittest

import numpy as np
import pytest
from numpy.random import default_rng

from flex.data import FlexDataDistribution, FlexDataObject, FlexDatasetConfig


class TestFlexDataDistribution(unittest.TestCase):
    def test_init_method_does_not_work(self):
        with pytest.raises(AssertionError):
            FlexDataDistribution()

    def test_client_names(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        config = FlexDatasetConfig(n_clients=3, client_names=["Juan", "Pepe", 2])
        flex_dataset = FlexDataDistribution.from_config(fcd, config)
        assert "Juan" in flex_dataset
        assert "Pepe" in flex_dataset
        assert 2 in flex_dataset

    def test_nclients(self):
        rng = default_rng()
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = rng.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        config = FlexDatasetConfig(n_clients=2)
        flex_dataset = FlexDataDistribution.from_config(fcd, config)
        assert len(flex_dataset) == config.n_clients
        assert len(flex_dataset[0]) == len(flex_dataset[1])
        assert len(flex_dataset[0]) + len(flex_dataset[1]) == len(fcd)

    def test_weights(self):
        rng = default_rng()
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = rng.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        config = FlexDatasetConfig(n_clients=2, weights=[1, 1], replacement=False)
        flex_dataset = FlexDataDistribution.from_config(fcd, config)
        assert len(flex_dataset) == config.n_clients
        assert len(flex_dataset[0]) == len(flex_dataset[1])
        assert len(np.unique(flex_dataset[0].y_data)) == 2
        assert len(np.unique(flex_dataset[1].y_data)) == 2

    def test_empty_weights(self):
        rng = default_rng()
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = rng.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        config = FlexDatasetConfig(n_clients=2, weights=None, replacement=False)
        flex_dataset = FlexDataDistribution.from_config(fcd, config)
        assert len(flex_dataset) == config.n_clients
        assert len(flex_dataset[0]) == len(flex_dataset[1])
        assert len(np.unique(flex_dataset[0].y_data)) == 2
        assert len(np.unique(flex_dataset[1].y_data)) == 2

    def test_classes_per_client_int_no_weights_no_replacement(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.concatenate((np.zeros(10), np.ones(10)))
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        config = FlexDatasetConfig(n_clients=2, classes_per_client=1, replacement=False)
        flex_dataset = FlexDataDistribution.from_config(fcd, config)
        assert len(flex_dataset) == config.n_clients
        assert len(flex_dataset[0]) + len(flex_dataset[1]) == len(fcd)
        assert len(np.unique(flex_dataset[0].y_data)) == 1
        assert len(np.unique(flex_dataset[1].y_data)) == 1

    def test_classes_per_client_int_no_weights_with_replacements(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.concatenate((np.zeros(10), np.ones(10)))
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        config = FlexDatasetConfig(n_clients=2, classes_per_client=1, replacement=True)
        flex_dataset = FlexDataDistribution.from_config(fcd, config)
        assert len(flex_dataset) == config.n_clients
        assert len(flex_dataset[0]) + len(flex_dataset[1]) == len(fcd)
        assert len(np.unique(flex_dataset[0].y_data)) == 1
        assert len(np.unique(flex_dataset[1].y_data)) == 1

    def test_classes_per_client_int_with_weigths_no_replacement(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.concatenate((np.zeros(10), np.ones(10)))
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        config = FlexDatasetConfig(
            n_clients=2, classes_per_client=1, weights=[0.25, 0.5], replacement=False
        )
        flex_dataset = FlexDataDistribution.from_config(fcd, config)
        assert len(flex_dataset) == config.n_clients
        if np.unique(flex_dataset[0].y_data) == np.unique(flex_dataset[1].y_data):
            assert len(flex_dataset[0]) + len(flex_dataset[1]) == 6
        else:
            assert len(flex_dataset[0]) + len(flex_dataset[1]) == 7
        assert len(np.unique(flex_dataset[0].y_data)) == 1
        assert len(np.unique(flex_dataset[1].y_data)) == 1

    def test_classes_per_client_int_with_weigths_with_replacement(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.concatenate((np.zeros(10), np.ones(10)))
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        config = FlexDatasetConfig(
            seed=1,
            n_clients=2,
            classes_per_client=1,
            weights=[0.25, 0.5],
            replacement=True,
        )
        flex_dataset = FlexDataDistribution.from_config(fcd, config)
        assert len(flex_dataset) == config.n_clients
        assert len(flex_dataset[0]) + len(flex_dataset[1]) == int(
            sum(np.floor(np.array(config.weights) * 10))
        )
        assert len(np.unique(flex_dataset[0].y_data)) == 1
        assert len(np.unique(flex_dataset[1].y_data)) == 1

    def test_classes_per_client_tuple_with_weights_no_replacement(self):
        X_data = np.random.rand(150).reshape([30, 5])
        y_data = np.concatenate(
            (
                np.zeros(5),
                np.ones(5),
                2 * np.ones(5),
                3 * np.ones(5),
                4 * np.ones(5),
                5 * np.ones(5),
            )
        )
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        config = FlexDatasetConfig(
            seed=1,
            n_clients=2,
            classes_per_client=(2, 3),
            weights=[0.25, 1],
            replacement=False,
        )

        flex_dataset = FlexDataDistribution.from_config(fcd, config)
        assert len(flex_dataset) == config.n_clients
        for k in flex_dataset:
            assert len(np.unique(flex_dataset[k].y_data)) <= 3
            assert len(np.unique(flex_dataset[k].y_data)) >= 2

    def test_classes_per_client_tuple_with_weights_with_replacement(self):
        X_data = np.random.rand(150).reshape([30, 5])
        y_data = np.concatenate(
            (
                np.zeros(5),
                np.ones(5),
                2 * np.ones(5),
                3 * np.ones(5),
                4 * np.ones(5),
                5 * np.ones(5),
            )
        )
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        config = FlexDatasetConfig(
            seed=1,
            n_clients=2,
            classes_per_client=(2, 3),
            weights=[0.25, 1],
            replacement=True,
        )

        flex_dataset = FlexDataDistribution.from_config(fcd, config)
        assert len(flex_dataset) == config.n_clients
        for k in flex_dataset:
            assert len(np.unique(flex_dataset[k].y_data)) <= 3
            assert len(np.unique(flex_dataset[k].y_data)) >= 2

    def test_classes_per_client_arr_no_weights_no_replacement(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        config = FlexDatasetConfig(
            seed=2,
            n_clients=2,
            classes_per_client=[0, 1],
            weights=None,
            replacement=False,
        )
        flex_dataset = FlexDataDistribution.from_config(fcd, config)
        assert np.unique(flex_dataset[0].y_data)[0] == 0
        assert np.unique(flex_dataset[1].y_data)[0] == 1

    def test_classes_per_client_arr_no_weights_with_replacement(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        config = FlexDatasetConfig(
            seed=2,
            n_clients=2,
            classes_per_client=[0, [0, 1]],
            weights=None,
            replacement=True,
        )
        flex_dataset = FlexDataDistribution.from_config(fcd, config)
        assert len(flex_dataset) == config.n_clients
        assert np.unique(flex_dataset[0].y_data)[0] == 0
        assert set(np.unique(flex_dataset[1].y_data)) == {0, 1}

    def test_classes_per_client_arr_with_weights_with_replacement(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        config = FlexDatasetConfig(
            seed=2,
            n_clients=2,
            classes_per_client=[0, [1, 0]],
            weights=[0.25, 0.5],
            replacement=True,
        )
        flex_dataset = FlexDataDistribution.from_config(fcd, config)
        assert len(flex_dataset) == config.n_clients
        assert np.unique(flex_dataset[0].y_data)[0] == 0
        assert set(np.unique(flex_dataset[1].y_data)) == {0, 1}

    def test_classes_per_client_arr_with_weights_no_replacement(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        config = FlexDatasetConfig(
            seed=2,
            n_clients=2,
            classes_per_client=[0, 1],
            weights=[0.25, 0.5],
            replacement=False,
        )
        flex_dataset = FlexDataDistribution.from_config(fcd, config)
        assert len(flex_dataset) == config.n_clients
        assert np.unique(flex_dataset[0].y_data)[0] == 0
        assert np.unique(flex_dataset[1].y_data)[0] == 1

    # Feature split testing
    def test_featutes_per_client_int(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        config = FlexDatasetConfig(
            seed=2, n_clients=2, features_per_client=3, replacement=True
        )
        flex_dataset = FlexDataDistribution.from_config(fcd, config)
        assert len(flex_dataset) == config.n_clients
        for k in flex_dataset:
            assert flex_dataset[k].X_data.shape[1] == 3

    def test_featutes_per_client_tuple(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        config = FlexDatasetConfig(
            seed=2, n_clients=2, features_per_client=(3, 5), replacement=True
        )
        flex_dataset = FlexDataDistribution.from_config(fcd, config)
        assert len(flex_dataset) == config.n_clients
        for k in flex_dataset:
            assert flex_dataset[k].X_data.shape[1] <= 5
            assert flex_dataset[k].X_data.shape[1] >= 3

    def test_featutes_per_client_arr(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        config = FlexDatasetConfig(
            seed=2, n_clients=2, features_per_client=[[2, 4], [1, 3]], replacement=True
        )
        flex_dataset = FlexDataDistribution.from_config(fcd, config)
        assert len(flex_dataset) == config.n_clients
        assert flex_dataset[0].X_data.shape[1] == 2
        assert flex_dataset[1].X_data.shape[1] == 2
