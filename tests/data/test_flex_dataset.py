import unittest

import numpy as np
import pytest

from flex.data.flex_dataset import FlexDataObject, FlexDataset
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


class TestFlexDataObject(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_simple_flex_data_object(self, fcd):
        self._fcd = fcd

    def test_X_data_property(self):
        X_data = np.random.rand(100).reshape([20, 5])
        self._fcd.X_data = X_data
        assert np.array_equal(X_data, self._fcd.X_data)

    def test_y_data_property(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        self._fcd.X_data = X_data
        self._fcd.y_data = y_data
        assert np.array_equal(y_data, self._fcd.y_data)
        assert np.array_equal(y_data, self._fcd.y_data)

    def test_X_names_property_valid_X_names(self):
        X_names = np.array([f"x{i}" for i in range(self._fcd.X_data.shape[1])])
        self._fcd.X_names = X_names
        assert np.array_equal(X_names, self._fcd.X_names)
        assert self._fcd.X_data.shape[0] == self._fcd.y_data.shape[0]

    def test_X_names_property_invalid_X_names(self):
        X_names = [f"x{i}" for i in range(self._fcd.X_data.shape[1] + 4)]
        with pytest.raises(Exception):
            self._fcd.X_names = X_names
            self._fcd.validate()

    def test_X_names_property_setter(self):
        X_names = [f"x{i}" for i in range(self._fcd.X_data.shape[1])]
        self._fcd.X_names = X_names
        assert np.array_equal(X_names, self._fcd.X_names)

    def test_y_names_property_valid_y_names(self):
        y_names = [f"class_{c}" for c in np.unique(self._fcd.y_data)]
        self._fcd.y_names = y_names
        assert np.array_equal(y_names, self._fcd.y_names)

    def test_empty_y_names_y_data(self):
        assert self._fcd.y_names is None

    def test_y_names_property_setter(self):
        y_names = [f"class_{c}" for c in np.unique(self._fcd.y_data)]
        self._fcd.y_names = y_names
        assert np.array_equal(y_names, self._fcd.y_names)

    def test_len_property(self):
        X_data = np.random.rand(100).reshape([20, 5])
        self._fcd.X_data = X_data
        assert len(self._fcd) == len(X_data)

    def test_getitem_property(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        for x, y, (x_bis, y_bis) in zip(X_data, y_data, fcd):
            assert np.array_equal(x, x_bis)
            assert y == y_bis
        fcd = FlexDataObject(X_data=X_data, y_data=None)
        for x, (x_bis, y_bis) in zip(X_data, fcd):
            assert np.array_equal(x, x_bis)
            assert y_bis is None

    def test_validate_correct_object(self):
        self._fcd.validate()

    def test_validate_invalid_object(self):
        X_names = [f"x{i}" for i in range(self._fcd.X_data.shape[1] - 1)]
        self._fcd.X_names = X_names
        with pytest.raises(ValueError):
            self._fcd.validate()
        X_names = [f"x{i}" for i in range(self._fcd.X_data.shape[1])]
        y_names = [f"class_{c}" for c in range(len(np.unique(self._fcd.y_data)) - 1)]
        self._fcd.X_names = X_names
        self._fcd.y_names = y_names
        with pytest.raises(ValueError):
            self._fcd.validate()

    def test_len_X_data_differs_len_y_data(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 19)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        with pytest.raises(ValueError):
            fcd.validate()
        y_data = np.random.choice(2, 30)
        fcd.y_data = y_data
        with pytest.raises(ValueError):
            fcd.validate()

    def test_y_data_multidimensional(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.randint(0, 2, size=(20, 4))
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        with pytest.raises(ValueError):
            fcd.validate()


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
        new_fld = self._fld.normalize(num_proc=1)
        assert all(
            not np.array_equal(client_orig.X_data, client_mod.X_data)
            for client_orig, client_mod in zip(self._fld.values(), new_fld.values())
        )

    def test_map_method(self):
        new_fld = self._fld.map(func=normalize)
        assert all(
            not np.array_equal(client_orig.X_data, client_mod.X_data)
            for client_orig, client_mod in zip(self._fld.values(), new_fld.values())
        )

    def test_map_func_none(self):
        with pytest.raises(ValueError):
            self._fld.map(func=None)

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

    def test_one_hot_encoding(self):
        new_fld = self._fld.one_hot_encoding(n_classes=2)
        assert all(client.y_data.shape[1] == 2 for client_id, client in new_fld.items())

    def test_all_clients_in_flex_dataset_when_mapping_func(self):
        client_ids = ["client_1", "client_84"]
        with pytest.raises(ValueError):
            self._fld.map(func=normalize, num_proc=10, clients_ids=client_ids)
