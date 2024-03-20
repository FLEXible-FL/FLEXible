"""
Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import unittest

import numpy as np
import pytest

from flex.data.dataset import Dataset
from flex.data.fed_dataset import FedDataset
from flex.data.preprocessing_utils import normalize


@pytest.fixture(name="fld")
def fixture_flex_dataset():
    """Function that returns a FlexDataset provided as example to test functions.

    Returns:
        FedDataset: A FlexDataset generated randomly
    """
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    fcd = Dataset.from_array(X_data, y_data)
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    fcd1 = Dataset.from_array(X_data, y_data)
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    fcd2 = Dataset.from_array(X_data, y_data)
    return FedDataset({"client_1": fcd, "client_2": fcd1, "client_3": fcd2})


@pytest.fixture(name="fcd")
def fixture_simple_fex_data_object():
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    return Dataset.from_array(X_data, y_data)


class TestFlexDataset(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_flex_dataset(self, fld):
        self._fld = fld

    def test_get_method(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = Dataset.from_array(X_data, y_data)
        flex_data = FedDataset()
        flex_data["client_1"] = fcd
        assert flex_data["client_1"] == fcd
        assert flex_data.get("client_2") is None

    def test_normalize_method(self):
        new_fld = self._fld.normalize()
        assert all(
            not np.array_equal(
                client_orig.X_data.to_numpy(), client_mod.X_data.to_numpy()
            )
            for client_orig, client_mod in zip(self._fld.values(), new_fld.values())
        )

    def test_one_hot_encoding(self):
        new_fld = self._fld.one_hot_encoding(n_labels=2)
        assert all(
            client.y_data.to_numpy().shape[1] == 2 for _, client in new_fld.items()
        )

    def test_map_method(self):
        new_fld = self._fld.apply(func=normalize)
        assert all(
            not np.array_equal(client_orig.X_data, client_mod.X_data)
            for client_orig, client_mod in zip(self._fld.values(), new_fld.values())
        )

    def test_map_func_from_outside(self):
        def dummy_func(data, **kwargs):
            return data

        new_fld = self._fld.apply(func=dummy_func, num_proc=2)
        assert all(
            np.array_equal(client_orig.X_data, client_mod.X_data)
            for client_orig, client_mod in zip(self._fld.values(), new_fld.values())
        )

    def test_proprocessing_custom_func_more_processes_than_clients(self):
        new_fld = self._fld.apply(func=normalize, num_proc=10)
        assert all(
            not np.array_equal(client_orig.X_data, client_mod.X_data)
            for client_orig, client_mod in zip(self._fld.values(), new_fld.values())
        )

    def test_chosen_clients_custom_func(self):
        chosen_clients = ["client_1", "client_2"]
        new_fld = self._fld.apply(func=normalize, num_proc=10, node_ids=chosen_clients)
        assert any(
            np.array_equal(client_orig.X_data, client_mod.X_data)
            for client_orig, client_mod in zip(self._fld.values(), new_fld.values())
        )

    def test_chosen_clients_normalize_data_func(self):
        chosen_clients = ["client_1", "client_2"]
        new_fld = self._fld.normalize(num_proc=10, node_ids=chosen_clients)
        assert any(
            np.array_equal(client_orig.X_data, client_mod.X_data)
            for client_orig, client_mod in zip(self._fld.values(), new_fld.values())
        )

    def test_all_clients_in_flex_dataset_when_mapping_func(self):
        client_ids = ["client_1", "client_84"]
        with pytest.raises(ValueError):
            self._fld.apply(func=normalize, num_proc=10, node_ids=client_ids)

    def test_map_func_executes_secuential(self):
        chosen_clients = ["client_1"]
        new_fld = self._fld.normalize(num_proc=1, node_ids=chosen_clients)
        assert any(
            np.array_equal(client_orig.X_data, client_mod.X_data)
            for client_orig, client_mod in zip(self._fld.values(), new_fld.values())
        )

    def test_map_recieves_one_client_as_str_correct(self):
        chosen_clients = "client_1"
        new_fld = self._fld.normalize(num_proc=1, node_ids=chosen_clients)
        assert any(
            np.array_equal(client_orig.X_data, client_mod.X_data)
            for client_orig, client_mod in zip(self._fld.values(), new_fld.values())
        )

    def test_map_recieves_one_client_as_str_fails(self):
        client_ids = "client_8232"
        with pytest.raises(ValueError):
            self._fld.apply(func=normalize, num_proc=1, node_ids=client_ids)
