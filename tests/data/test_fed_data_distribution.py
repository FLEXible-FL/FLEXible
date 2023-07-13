import unittest
from math import isclose

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

from flex.data import Dataset, FedDataDistribution, FedDataset, FedDatasetConfig
from flex.datasets import load


@pytest.fixture(name="fcd")
def fixture_simple_fex_data_object():
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    return Dataset(X_data=X_data, y_data=y_data)


@pytest.fixture(name="fcd_ones_zeros")
def fixture_simple_fex_data_object_ones_zeros():
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.concatenate((np.zeros(10), np.ones(10)))
    return Dataset(X_data=X_data, y_data=y_data)


@pytest.fixture(name="fcd_multiple_classes")
def fixture_simple_fex_data_object_multiple_classes():
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
    return Dataset(X_data=X_data, y_data=y_data)


class TestFlexDataDistribution(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_iris_dataset(self):
        iris = load_iris()
        self._iris = Dataset(X_data=iris.data, y_data=iris.target)

    @pytest.fixture(autouse=True)
    def _fixture_simple_flex_data_object(self, fcd):
        self._fcd = fcd

    @pytest.fixture(autouse=True)
    def _fixture_simple_flex_data_object_ones_zeros(self, fcd_ones_zeros):
        self._fcd_ones_zeros = fcd_ones_zeros

    @pytest.fixture(autouse=True)
    def _fixture_simple_flex_data_object_multiple_classes(self, fcd_multiple_classes):
        self._fcd_multiple_classes = fcd_multiple_classes

    def test_init_method_does_not_work(self):
        with pytest.raises(AssertionError):
            FedDataDistribution()

    def test_client_names(self):
        config = FedDatasetConfig(n_clients=3, client_names=["Juan", "Pepe", 2])
        flex_dataset = FedDataDistribution.from_config(self._fcd, config)
        assert "Juan" in flex_dataset
        assert "Pepe" in flex_dataset
        assert 2 in flex_dataset

    def test_client_names_bis(self):
        config = FedDatasetConfig(n_clients=3, client_names=["Juan", "Pepe", 2])
        flex_dataset = FedDataDistribution.from_config(self._fcd, config)
        assert "Juan" in flex_dataset
        assert "Pepe" in flex_dataset
        assert 2 in flex_dataset

    def test_nclients(self):
        config = FedDatasetConfig(n_clients=2)
        flex_dataset = FedDataDistribution.from_config(self._fcd, config)
        assert len(flex_dataset) == config.n_clients
        assert len(flex_dataset[0]) == len(flex_dataset[1])
        assert len(flex_dataset[0]) + len(flex_dataset[1]) == len(self._fcd)

    def test_weights(self):
        config = FedDatasetConfig(n_clients=2, weights=[1, 1], replacement=False)
        flex_dataset = FedDataDistribution.from_config(self._fcd, config)
        assert len(flex_dataset) == config.n_clients
        assert len(flex_dataset[0]) == len(flex_dataset[1])
        assert len(np.unique(flex_dataset[0].y_data)) == 2
        assert len(np.unique(flex_dataset[1].y_data)) == 2

    def test_empty_weights(self):
        config = FedDatasetConfig(n_clients=2, weights=None, replacement=False)
        flex_dataset = FedDataDistribution.from_config(self._fcd, config)
        assert len(flex_dataset) == config.n_clients
        assert len(flex_dataset[1]) == len(flex_dataset[0])
        assert len(np.unique(flex_dataset[1].y_data)) == 2
        assert len(np.unique(flex_dataset[0].y_data)) == 2

    def test_classes_per_client_int_no_weights_no_replacement(self):
        config = FedDatasetConfig(
            seed=1, n_clients=2, classes_per_client=1, replacement=False
        )
        flex_dataset = FedDataDistribution.from_config(self._fcd_ones_zeros, config)
        assert len(flex_dataset) == config.n_clients
        assert len(flex_dataset[0]) + len(flex_dataset[1]) == len(self._fcd_ones_zeros)
        assert len(np.unique(flex_dataset[0].y_data)) == 1
        assert len(np.unique(flex_dataset[1].y_data)) == 1

    def test_classes_per_client_int_no_weights_with_replacements(self):
        config = FedDatasetConfig(n_clients=2, classes_per_client=1, replacement=True)
        flex_dataset = FedDataDistribution.from_config(self._fcd_ones_zeros, config)
        assert len(flex_dataset) == config.n_clients
        assert len(flex_dataset[0]) + len(flex_dataset[1]) == len(self._fcd_ones_zeros)
        assert len(np.unique(flex_dataset[0].y_data)) == 1
        assert len(np.unique(flex_dataset[1].y_data)) == 1

    def test_classes_per_client_int_with_weigths_no_replacement(self):
        config = FedDatasetConfig(
            n_clients=2, classes_per_client=1, weights=[0.25, 0.5], replacement=False
        )
        flex_dataset = FedDataDistribution.from_config(self._fcd_ones_zeros, config)
        assert len(flex_dataset) == config.n_clients
        assert len(flex_dataset[0]) == 2
        assert len(flex_dataset[1]) == 5
        assert len(np.unique(flex_dataset[0].y_data)) == 1
        assert len(np.unique(flex_dataset[1].y_data)) == 1

    def test_classes_per_client_int_with_weigths_with_replacement(self):
        config = FedDatasetConfig(
            seed=1,
            n_clients=2,
            classes_per_client=1,
            weights=[0.25, 0.5],
            replacement=True,
        )
        flex_dataset = FedDataDistribution.from_config(self._fcd_ones_zeros, config)
        assert len(flex_dataset) == config.n_clients
        assert len(flex_dataset[0]) + len(flex_dataset[1]) == int(
            sum(np.floor(np.array(config.weights) * 10))
        )
        assert len(np.unique(flex_dataset[0].y_data)) == 1
        assert len(np.unique(flex_dataset[1].y_data)) == 1

    def test_classes_per_client_tuple_with_weights_no_replacement(self):
        config = FedDatasetConfig(
            seed=1,
            n_clients=2,
            classes_per_client=(1, 2),
            weights=[0.5, 0.5],
            replacement=False,
        )

        flex_dataset = FedDataDistribution.from_config(
            self._fcd_multiple_classes, config
        )
        assert len(flex_dataset) == config.n_clients
        for k in flex_dataset:
            assert len(np.unique(flex_dataset[k].y_data)) <= 2
            assert len(np.unique(flex_dataset[k].y_data)) >= 1

    def test_classes_per_client_tuple_with_weights_with_replacement(self):
        config = FedDatasetConfig(
            seed=1,
            n_clients=2,
            classes_per_client=(2, 3),
            weights=[0.5, 1],
            replacement=True,
        )

        flex_dataset = FedDataDistribution.from_config(
            self._fcd_multiple_classes, config
        )
        assert len(flex_dataset) == config.n_clients
        for k in flex_dataset:
            assert len(np.unique(flex_dataset[k].y_data)) <= 3
            assert len(np.unique(flex_dataset[k].y_data)) >= 2

    def test_classes_per_client_arr_no_weights_no_replacement(self):
        config = FedDatasetConfig(
            seed=2,
            n_clients=2,
            client_names=["client_0", "client_1"],
            classes_per_client=[[0], [1]],
            weights=None,
            replacement=False,
        )
        flex_dataset = FedDataDistribution.from_config(self._iris, config)
        assert np.unique(flex_dataset["client_0"].y_data)[0] == 0
        assert np.unique(flex_dataset["client_1"].y_data)[0] == 1

    def test_classes_per_client_arr_no_weights_with_replacement(self):
        config = FedDatasetConfig(
            seed=2,
            n_clients=2,
            client_names=["client_0", "client_1"],
            classes_per_client=[[0], [0, 1]],
            weights=None,
            replacement=True,
        )
        flex_dataset = FedDataDistribution.from_config(self._iris, config)
        assert len(flex_dataset) == config.n_clients
        assert np.unique(flex_dataset["client_0"].y_data)[0] == 0
        assert set(np.unique(flex_dataset["client_1"].y_data)) == {0, 1}

    def test_classes_per_client_arr_with_weights_with_replacement(self):
        config = FedDatasetConfig(
            seed=2,
            n_clients=2,
            client_names=["client_0", "client_1"],
            classes_per_client=[[0], [1, 0]],
            weights=[0.25, 0.5],
            replacement=True,
        )
        flex_dataset = FedDataDistribution.from_config(self._iris, config)
        assert len(flex_dataset) == config.n_clients
        assert np.unique(flex_dataset["client_0"].y_data)[0] == 0
        assert set(np.unique(flex_dataset["client_1"].y_data)) == {0, 1}

    def test_classes_per_client_arr_with_weights_no_replacement(self):
        config = FedDatasetConfig(
            seed=2,
            n_clients=2,
            client_names=["client_0", "client_1"],
            classes_per_client=[[0], [1]],
            weights=[0.25, 0.5],
            replacement=False,
        )
        flex_dataset = FedDataDistribution.from_config(self._iris, config)
        assert len(flex_dataset) == config.n_clients
        assert np.unique(flex_dataset["client_0"].y_data)[0] == 0
        assert np.unique(flex_dataset["client_1"].y_data)[0] == 1

    # Feature split testing
    def test_featutes_per_client_int(self):
        config = FedDatasetConfig(
            seed=2, n_clients=2, features_per_client=3, replacement=True
        )
        flex_dataset = FedDataDistribution.from_config(self._iris, config)
        assert len(flex_dataset) == config.n_clients
        for k in flex_dataset:
            assert flex_dataset[k].X_data.shape[1] == 3

    def test_featutes_per_client_tuple(self):
        min_features = 1
        max_features = 3
        config = FedDatasetConfig(
            seed=2,
            n_clients=2,
            features_per_client=(min_features, max_features),
            replacement=True,
        )
        flex_dataset = FedDataDistribution.from_config(self._iris, config)
        assert len(flex_dataset) == config.n_clients
        for k in flex_dataset:
            assert flex_dataset[k].X_data.shape[1] <= max_features
            assert flex_dataset[k].X_data.shape[1] >= min_features

    def test_featutes_per_client_arr(self):
        config = FedDatasetConfig(
            seed=2,
            n_clients=2,
            client_names=["client_0", "client_1"],
            features_per_client=[[1, 3], [0, 2]],
            replacement=True,
        )
        flex_dataset = FedDataDistribution.from_config(self._iris, config)
        assert len(flex_dataset) == config.n_clients
        assert flex_dataset["client_0"].X_data.shape[1] == 2
        assert flex_dataset["client_1"].X_data.shape[1] == 2

    def test_iid_distribution(self):
        n_clients = 2
        flex_dataset = FedDataDistribution.iid_distribution(self._iris, n_clients)
        assert len(flex_dataset) == n_clients
        assert len(flex_dataset[0]) == len(flex_dataset[1])
        assert len(flex_dataset[0]) + len(flex_dataset[1]) == len(self._iris)

    def test_single_feature_data(self):
        single_feature_dataset = Dataset(self._iris.X_data[:, 0], self._iris.y_data)
        federated_iris = FedDataDistribution.iid_distribution(
            centralized_data=single_feature_dataset
        )
        assert len(federated_iris[0].X_data.shape) == 1

    def test_indexes_per_client(self):
        indexes = [[1, 3], [0, 2]]
        config = FedDatasetConfig(
            n_clients=len(indexes), replacement=False, indexes_per_client=indexes
        )
        federated_iris = FedDataDistribution.from_config(self._iris, config)
        assert all(
            np.array_equal(federated_iris[client].X_data, self._iris.X_data[idx])
            and np.array_equal(federated_iris[client].y_data, self._iris.y_data[idx])
            for idx, client in zip(indexes, federated_iris)
        )

    def test_from_clustering_func(self):
        n_clients = 10
        kmeans = KMeans(n_clusters=n_clients, random_state=0).fit(self._iris.X_data)
        federated_iris = FedDataDistribution.from_clustering_func(
            self._iris, clustering_func=lambda x, _: kmeans.predict(x.reshape(1, -1))[0]
        )
        assert len(federated_iris) == n_clients
        assert all(
            self._iris.X_data[idx] in federated_iris[client].X_data
            for idx, client in enumerate(kmeans.labels_)
        )

    def test_weight_per_classes_random_assigment(self):
        classes = np.unique(self._iris.y_data)
        config = FedDatasetConfig(
            seed=2,
            n_clients=len(classes),
            weights_per_class=[[1, 2, 3, 4], [0, 0, 0, 1], [0, 1, 0, 0]],
            replacement=False,
        )
        flex_dataset = FedDataDistribution.from_config(self._iris, config)
        clients = list(flex_dataset.keys())
        assert all(
            sum(flex_dataset[clients[0]].y_data == i)
            != sum(flex_dataset[clients[1]].y_data == i)
            for i in classes
        )

    def test_weight_per_class_alone_w_replacement(self):
        classes = np.unique(self._iris.y_data)
        config = FedDatasetConfig(
            seed=2,
            n_clients=2,
            weights_per_class=np.ones((2, len(classes))),
            replacement=True,
        )
        flex_dataset = FedDataDistribution.from_config(self._iris, config)
        clients = list(flex_dataset.keys())
        assert all(
            sum(flex_dataset[clients[0]].y_data == i)
            == sum(flex_dataset[clients[1]].y_data == i)
            for i in classes
        )

    def test_weight_per_class_alone_without_replacement(self):
        classes = np.unique(self._iris.y_data)
        config = FedDatasetConfig(
            seed=2,
            n_clients=2,
            weights_per_class=np.ones((2, len(classes))),
            replacement=False,
        )
        flex_dataset = FedDataDistribution.from_config(self._iris, config)
        clients = list(flex_dataset.keys())
        assert all(
            sum(flex_dataset[clients[0]].y_data == i)
            == sum(flex_dataset[clients[1]].y_data == i)
            for i in classes
        )
        assert all(
            np.sum(flex_dataset[clients[0]].X_data[i])
            != np.sum(flex_dataset[clients[1]].X_data[i])
            for i in range(len(flex_dataset))
        )

    def test_from_torchtext_dataset(self):
        from torchtext.datasets import AG_NEWS

        data = AG_NEWS(split="test")
        config = FedDatasetConfig(
            seed=0,
            n_clients=2,
            replacement=False,
            client_names=["client_0", "client_1"],
        )
        flex_dataset = FedDataDistribution.from_config_with_torchtext_dataset(
            data, config
        )
        assert len(flex_dataset) == config.n_clients
        assert len(flex_dataset["client_0"]) == len(flex_dataset["client_1"])

    def test_from_tfds_image_dataset(self):
        import tensorflow_datasets as tfds

        # With batch_size -1
        def build_data_and_check(data, config):
            flex_dataset = FedDataDistribution.from_config_with_tfds_image_dataset(
                data, config
            )
            assert len(flex_dataset) == config.n_clients
            assert len(flex_dataset["client_0"]) == len(flex_dataset["client_1"])

        other_options = {
            "split": "test",
            "shuffle_files": True,
            "as_supervised": True,
            "batch_size": -1,
        }
        ds_name = "cifar10"
        data = tfds.load(ds_name, **other_options)
        config = FedDatasetConfig(
            seed=0,
            n_clients=2,
            replacement=False,
            client_names=["client_0", "client_1"],
        )
        # With batch_size = 20
        other_options["batch_size"] = 20
        data = tfds.load(ds_name, **other_options)
        build_data_and_check(data, config)

    def test_from_tfds_image_dataset_without_batchsize(self):
        import tensorflow_datasets as tfds

        def build_data_and_check(data, config):
            flex_dataset = FedDataDistribution.from_config_with_tfds_image_dataset(
                data, config
            )
            assert len(flex_dataset) == config.n_clients
            assert len(flex_dataset["client_0"]) == len(flex_dataset["client_1"])

        other_options = {
            "split": "test",
            "shuffle_files": True,
            "as_supervised": True,
            "batch_size": -1,
        }
        ds_name = "cifar10"
        data = tfds.load(ds_name, **other_options)
        config = FedDatasetConfig(
            seed=0,
            n_clients=2,
            replacement=False,
            client_names=["client_0", "client_1"],
        )
        data = tfds.load(ds_name, **other_options)
        build_data_and_check(data, config)

    def test_from_tfds_text_dataset(self):
        import tensorflow_datasets as tfds

        def build_data_and_check(data, config, X_columns, labels):
            flex_dataset = FedDataDistribution.from_config_with_tfds_text_dataset(
                data, config, X_columns, labels
            )
            assert len(flex_dataset) == config.n_clients
            assert len(flex_dataset["client_0"]) == len(flex_dataset["client_1"])

        other_options = {"split": "test"}
        X_columns = ["title", "description"]
        labels = ["label"]
        data = tfds.load("ag_news_subset", **other_options)
        config = FedDatasetConfig(
            seed=0,
            n_clients=2,
            replacement=False,
            client_names=["client_0", "client_1"],
        )
        # With batch_size
        other_options["batch_size"] = 20
        data = tfds.load("ag_news_subset", **other_options)
        build_data_and_check(data, config, X_columns, labels)

    def test_from_tfds_text_dataset_without_batchsize(self):
        import tensorflow_datasets as tfds

        def build_data_and_check(data, config, X_columns, labels):
            flex_dataset = FedDataDistribution.from_config_with_tfds_text_dataset(
                data, config, X_columns, labels
            )
            assert len(flex_dataset) == config.n_clients
            assert len(flex_dataset["client_0"]) == len(flex_dataset["client_1"])

        other_options = {"split": "test"}
        X_columns = ["title", "description"]
        labels = ["label"]
        data = tfds.load("ag_news_subset", **other_options)
        config = FedDatasetConfig(
            seed=0,
            n_clients=2,
            replacement=False,
            client_names=["client_0", "client_1"],
        )
        # Without batch_size
        data = tfds.load("ag_news_subset", **other_options)
        build_data_and_check(data, config, X_columns, labels)

    def test_from_torchvision_dataset(self):
        from torchvision.datasets import MNIST

        data = MNIST(root="./torch_datasets", train=False, download=True)
        config = FedDatasetConfig(
            seed=0,
            n_clients=2,
            replacement=False,
            client_names=["client_0", "client_1"],
        )
        flex_dataset = FedDataDistribution.from_config_with_torchvision_dataset(
            data, config
        )
        assert len(flex_dataset) == config.n_clients
        assert len(flex_dataset["client_0"]) == len(flex_dataset["client_1"])

    def test_from_torchvision_dataset_lazyly(self):
        from torchvision.datasets import Food101

        data = Food101(root="./torch_datasets", split="test", download=True)
        config = FedDatasetConfig(
            seed=0,
            n_clients=2,
            replacement=False,
            client_names=["client_0", "client_1"],
            classes_per_client=[[2, 3], [2]],
        )
        flex_dataset = FedDataDistribution.from_config_with_torchvision_dataset(
            data, config
        )
        assert len(flex_dataset) == config.n_clients
        assert not np.array_equal(
            flex_dataset["client_0"].X_data[1], flex_dataset["client_1"].X_data[1]
        )

    def test_from_huggingface_text_dataset(self):
        from datasets import load_dataset

        data = load_dataset("ag_news", split="test")
        X_columns = "text"
        label_columns = "label"
        config = FedDatasetConfig(
            seed=0,
            replacement=False,
            client_names=["client_0", "client_1"],
        )
        flex_dataset = FedDataDistribution.from_config_with_huggingface_dataset(
            data, config, X_columns, label_columns
        )
        assert len(flex_dataset) == config.n_clients
        assert len(flex_dataset["client_0"]) == len(flex_dataset["client_1"])
        assert (
            len(flex_dataset["client_0"]) + len(flex_dataset["client_1"])
            == data.num_rows
        )

    def test_loading_fedmnist_digits_using_from_config(self):
        fed_data, test_data = load("federated_emnist", return_test=True, split="digits")
        assert isinstance(fed_data, FedDataset)
        assert isinstance(test_data, Dataset)
        num_samples = [len(fed_data[i]) for i in fed_data]
        total_samples = np.sum(num_samples)
        std = np.std(num_samples)
        mean = np.mean(num_samples)
        users = len(fed_data)
        assert users == 3579
        assert total_samples == 240000
        assert isclose(mean, 67.05, abs_tol=1e-1)
        assert isclose(std, 11.17, abs_tol=1e-1)

    def test_loading_fedmnist_letters_using_from_config(self):
        fed_data, test_data = load(
            "federated_emnist", return_test=True, split="letters"
        )
        assert isinstance(fed_data, FedDataset)
        assert isinstance(test_data, Dataset)
        num_samples = [len(fed_data[i]) for i in fed_data]
        total_samples = np.sum(num_samples)
        std = np.std(num_samples)
        mean = np.mean(num_samples)
        users = len(fed_data)
        assert users == 3585
        assert total_samples == 124800
        assert isclose(mean, 34.81, abs_tol=1e-1)
        assert isclose(std, 21.85, abs_tol=1e-1)

    @pytest.mark.skip(
        reason="CelebA dataset from torchvision has a limited amount of downloads per day allowed"
    )
    def test_loading_fedceleba_using_from_config(self):
        fed_data, test_data = load("federated_celeba", return_test=True)
        assert isinstance(fed_data, FedDataset)
        assert isinstance(test_data, Dataset)
        num_samples = [len(fed_data[i]) for i in fed_data]
        total_samples = np.sum(num_samples)
        std = np.std(num_samples)
        mean = np.mean(num_samples)
        users = len(fed_data)
        assert users == 8192
        assert total_samples == 162770
        assert isclose(mean, 19.87, abs_tol=1e-1)
        assert isclose(std, 8.92, abs_tol=1e-1)

    def test_loading_fedsentiment_using_from_config(self):
        fed_data, test_data = load("federated_sentiment140", return_test=True)
        assert isinstance(fed_data, FedDataset)
        assert isinstance(test_data, Dataset)
        num_samples = [len(fed_data[i]) for i in fed_data]
        total_samples = np.sum(num_samples)
        std = np.std(num_samples)
        mean = np.mean(num_samples)
        users = len(fed_data)
        assert users == 659775
        assert total_samples == 1600000
        assert isclose(mean, 2.42, abs_tol=1e-1)
        assert isclose(std, 4.71, abs_tol=1e-1)

    def test_loading_fedshakespeare_using_from_config(self):
        fed_data, test_data = load("federated_shakespeare", return_test=True)
        assert isinstance(fed_data, FedDataset)
        assert isinstance(test_data, Dataset)
        num_samples = [len(fed_data[i]) for i in fed_data]
        total_samples = np.sum(num_samples)
        std = np.std(num_samples)
        mean = np.mean(num_samples)
        users = len(fed_data)
        assert users == 660
        assert total_samples == 3678451
        assert isclose(mean, 5573.41, abs_tol=1e-1)
        assert isclose(std, 6460.77, abs_tol=1e-1)

    def test_emnist_wrong_split_error(self):
        with pytest.raises(ValueError):
            load("emnist", split="weird")
