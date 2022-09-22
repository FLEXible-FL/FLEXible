import unittest

import numpy as np
import pytest

from flex.data.flex_data_object import FlexDataObject
from flex.data.flex_dataset import FlexDataset
from flex.data.flex_preprocessing_utils import normalize


@pytest.fixture(name="fld")
def fixture_flex_dataset():
    """Function that returns a FlexDataset provided as example to test functions.

    Returns:
        FlexDataset: A FlexDataset generated randomly
    """
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    fcd = FlexDataObject(X_data=X_data, y_data=y_data)
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    fcd1 = FlexDataObject(X_data=X_data, y_data=y_data)
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    fcd2 = FlexDataObject(X_data=X_data, y_data=y_data)
    return FlexDataset({"client_1": fcd, "client_2": fcd1, "client_3": fcd2})


@pytest.fixture(name="fcd")
def fixture_simple_fex_data_object():
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    return FlexDataObject(X_data=X_data, y_data=y_data)


class TestFlexDataset(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_flex_dataset(self, fld):
        self._fld = fld

    def test_get_method(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        flex_data = FlexDataset()
        flex_data["client_1"] = fcd
        assert flex_data["client_1"] == fcd
        assert flex_data.get("client_2") is None

    def test_normalize_method(self):
        new_fld = self._fld.normalize()
        assert all(
            not np.array_equal(client_orig.X_data, client_mod.X_data)
            for client_orig, client_mod in zip(self._fld.values(), new_fld.values())
        )

    def test_one_hot_encoding(self):
        new_fld = self._fld.one_hot_encoding(n_classes=2)
        assert all(client.y_data.shape[1] == 2 for _, client in new_fld.items())

    def test_map_method(self):
        new_fld = self._fld.map(func=normalize)
        assert all(
            not np.array_equal(client_orig.X_data, client_mod.X_data)
            for client_orig, client_mod in zip(self._fld.values(), new_fld.values())
        )

    def test_map_func_from_outside(self):
        def dummy_func(data, **kwargs):
            return data

        new_fld = self._fld.map(func=dummy_func, num_proc=2)
        assert all(
            np.array_equal(client_orig.X_data, client_mod.X_data)
            for client_orig, client_mod in zip(self._fld.values(), new_fld.values())
        )

    def test_proprocessing_custom_func_more_processes_than_clients(self):
        new_fld = self._fld.map(func=normalize, num_proc=10)
        assert all(
            not np.array_equal(client_orig.X_data, client_mod.X_data)
            for client_orig, client_mod in zip(self._fld.values(), new_fld.values())
        )

    def test_chosen_clients_custom_func(self):
        chosen_clients = ["client_1", "client_2"]
        new_fld = self._fld.map(func=normalize, num_proc=10, clients_ids=chosen_clients)
        assert any(
            np.array_equal(client_orig.X_data, client_mod.X_data)
            for client_orig, client_mod in zip(self._fld.values(), new_fld.values())
        )

    def test_chosen_clients_normalize_data_func(self):
        chosen_clients = ["client_1", "client_2"]
        new_fld = self._fld.normalize(num_proc=10, clients_ids=chosen_clients)
        assert any(
            np.array_equal(client_orig.X_data, client_mod.X_data)
            for client_orig, client_mod in zip(self._fld.values(), new_fld.values())
        )

    def test_all_clients_in_flex_dataset_when_mapping_func(self):
        client_ids = ["client_1", "client_84"]
        with pytest.raises(ValueError):
            self._fld.map(func=normalize, num_proc=10, clients_ids=client_ids)
