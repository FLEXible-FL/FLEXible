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
import numpy as np
import pytest

from flex.data.fed_dataset import Dataset
from flex.data.preprocessing_utils import normalize, one_hot_encoding


@pytest.fixture(name="fcd_ones")
def fixture_simple_fex_data_object_with_ones():
    X_data = np.ones(shape=(20, 5))
    y_data = np.random.choice(2, 20)
    return Dataset.from_array(X_data, y_data)


@pytest.fixture(name="fcd_zeros")
def fixture_simple_fex_data_object_with_zeros():
    X_data = np.zeros(shape=(20, 5))
    y_data = np.random.choice(2, 20)
    return Dataset.from_array(X_data, y_data)


def test_normalize_function(fcd_ones):
    X_data_normalized = normalize(fcd_ones).X_data.to_numpy()
    assert all(
        np.isclose(
            np.linalg.norm(X_data_normalized, axis=0),
            np.ones(X_data_normalized.shape[1]),
        )
    )
    assert fcd_ones.X_data.to_numpy().shape == X_data_normalized.shape


def test_normalize_func_norm_zero(fcd_zeros):
    X_data_normalized = normalize(fcd_zeros).X_data.to_numpy()
    assert np.array_equal(fcd_zeros.X_data.to_numpy(), X_data_normalized)
    assert fcd_zeros.X_data.to_numpy().shape == X_data_normalized.shape


def test_one_hot_encoding(fcd_ones):
    new_fcd = one_hot_encoding(fcd_ones, n_labels=2)
    assert new_fcd.y_data.to_numpy().shape[1] == 2
    assert len(new_fcd.y_data) == len(fcd_ones.y_data)


def test_one_hot_encoding_error(fcd_zeros):
    with pytest.raises(ValueError):
        one_hot_encoding(fcd_zeros)
