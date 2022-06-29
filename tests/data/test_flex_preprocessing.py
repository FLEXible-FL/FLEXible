import numpy as np
import pytest

from flex.data.flex_dataset import FlexDataObject
from flex.data.flex_preprocessing_utils import normalize, one_hot_encoding


@pytest.fixture(name="fcd_ones")
def fixture_simple_fex_data_object_with_ones():
    X_data = np.ones(shape=(20, 5))
    y_data = np.random.choice(2, 20)
    return FlexDataObject(X_data=X_data, y_data=y_data)


@pytest.fixture(name="fcd_zeros")
def fixture_simple_fex_data_object_with_zeros():
    X_data = np.zeros(shape=(20, 5))
    y_data = np.random.choice(2, 20)
    return FlexDataObject(X_data=X_data, y_data=y_data)


def test_normalize_function(fcd_ones):
    X_data_normalized = normalize(fcd_ones).X_data
    assert all(
        np.isclose(
            np.linalg.norm(X_data_normalized, axis=0),
            np.ones(X_data_normalized.shape[1]),
        )
    )
    assert fcd_ones.X_data.shape == X_data_normalized.shape


def test_normalize_func_norm_zero(fcd_zeros):
    X_data_normalized = normalize(fcd_zeros).X_data
    assert np.array_equal(fcd_zeros.X_data, X_data_normalized)
    assert fcd_zeros.X_data.shape == X_data_normalized.shape


def test_one_hot_encoding(fcd_ones):
    new_fcd = one_hot_encoding(fcd_ones, n_classes=2)
    assert new_fcd.y_data.shape[1] == 2
    assert len(new_fcd.y_data) == len(fcd_ones.y_data)


def test_one_hot_encoding_error(fcd_zeros):
    with pytest.raises(ValueError):
        one_hot_encoding(fcd_zeros)
