import numpy as np
import pytest

from flex.data.flex_dataset import FlexDataObject, FlexDataset
from flex.data.flex_preprocessing import (
    normalize,
    normalize_data_at_client,
    preprocessing_custom_func,
)


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
    return FlexDataset({"client_1": fcd, "client_2": fcd1, "client_3": fcd2})


def test_normalize_data_at_client(fld):
    new_fld = normalize_data_at_client(fld, processes=1)
    assert any(
        np.array_equal(client_orig.X_data, client_mod.X_data)
        for client_orig, client_mod in zip(fld.values(), new_fld.values())
    )


def test_preprocessing_custom_func(fld):
    new_fld = preprocessing_custom_func(fld, func=normalize)
    assert any(
        np.array_equal(client_orig.X_data, client_mod.X_data)
        for client_orig, client_mod in zip(fld.values(), new_fld.values())
    )


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


def test_preprocessing_custom_func_none(fld):
    with pytest.raises(ValueError):
        preprocessing_custom_func(fld, func=None)


def test_proprocessing_custom_func_more_processes_than_clients(fld):
    new_fld = preprocessing_custom_func(fld, func=normalize, processes=10)
    assert any(
        np.array_equal(client_orig.X_data, client_mod.X_data)
        for client_orig, client_mod in zip(fld.values(), new_fld.values())
    )


def test_chosen_clients_custom_func(fld):
    chosen_clients = ["client_1", "client_2"]
    new_fld = preprocessing_custom_func(
        fld, func=normalize, processes=10, clients_ids=chosen_clients
    )
    assert any(
        np.array_equal(client_orig.X_data, client_mod.X_data)
        for client_orig, client_mod in zip(fld.values(), new_fld.values())
    )


def test_chosen_clients_normalize_data_func(fld):
    chosen_clients = ["client_1", "client_2"]
    new_fld = normalize_data_at_client(fld, processes=10, clients_ids=chosen_clients)
    assert any(
        np.array_equal(client_orig.X_data, client_mod.X_data)
        for client_orig, client_mod in zip(fld.values(), new_fld.values())
    )


def test_normalize_func_norm_zero():
    X_data = np.zeros(shape=(20, 5))
    fcd = FlexDataObject(X_data=X_data)
    X_data_normalized = normalize(fcd).X_data
    assert np.array_equal(X_data, X_data_normalized)
    assert X_data.shape == X_data_normalized.shape
