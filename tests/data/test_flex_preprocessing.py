import unittest

import numpy as np
import pytest

from flex.data.flex_dataset import FlexDataObject, FlexDataset
from flex.data.flex_preprocessing import normalize_data_at_client, normalize


@pytest.fixture(name="fld")
def fixture_flex_dataset():
    """Function that returns a FlexDataset provided as example to test functions.

    Returns:
        FlexDataset: A FlexDataset generated randomly
    """
    X_data = np.random.rand(100).reshape([20, 5])
    fcd = FlexDataObject(X_data=X_data)
    X_data = np.random.rand(100).reshape([20, 5])
    fcd1 = FlexDataObject(X_data=X_data)
    X_data = np.random.rand(100).reshape([20, 5])
    fcd2 = FlexDataObject(X_data=X_data)
    return FlexDataset({'client_1': fcd, 'client_2': fcd1, 'client_3': fcd2})


def test_normalize_data_at_client(fld):
    print(f"FlexDataset no processed: {fld}")
    new_fld = normalize_data_at_client(fld, processes=1)
    print(f"FlexDataset processed: {new_fld}")
    assert fld != new_fld


def test_normalize_function():
    X_data = np.random.rand(100).reshape([20, 5])
    fcd = FlexDataObject(X_data=X_data)
    X_data_normalized = normalize(fcd)
    assert X_data != X_data_normalized
    assert X_data.shape == X_data_normalized.shape
