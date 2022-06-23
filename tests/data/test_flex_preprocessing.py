import numpy as np
import pytest

from flex.data.flex_dataset import FlexDataObject
from flex.data.flex_preprocessing import normalize, one_hot_encoding


def test_normalize_function():
    X_data = np.random.rand(100).reshape([20, 5])
    fcd = FlexDataObject(X_data=X_data)
    X_data_normalized = normalize(fcd).X_data
    assert not np.array_equal(X_data, X_data_normalized)
    assert X_data.shape == X_data_normalized.shape


def test_normalize_function_norm_zero():
    X_data = np.ones(shape=(20, 5))
    fcd = FlexDataObject(X_data=X_data)
    X_data_normalized = normalize(fcd).X_data
    assert not np.array_equal(X_data, X_data_normalized)
    assert X_data.shape == X_data_normalized.shape


def test_normalize_func_norm_zero():
    X_data = np.zeros(shape=(20, 5))
    fcd = FlexDataObject(X_data=X_data)
    X_data_normalized = normalize(fcd).X_data
    assert np.array_equal(X_data, X_data_normalized)
    assert X_data.shape == X_data_normalized.shape


def test_one_hot_encoding():
    X_data = np.zeros(shape=(20, 5))
    y_data = np.random.choice(2, 20)
    fcd = FlexDataObject(X_data=X_data, y_data=y_data)
    new_fcd = one_hot_encoding(fcd, n_classes=2)
    assert new_fcd.y_data.shape[1] == 2
    assert new_fcd.y_data.size == fcd.y_data.size


def test_one_hot_encoding_error():
    X_data = np.zeros(shape=(20, 5))
    y_data = np.random.choice(2, 20)
    fcd = FlexDataObject(X_data=X_data, y_data=y_data)
    with pytest.raises(ValueError):
        one_hot_encoding(fcd)
