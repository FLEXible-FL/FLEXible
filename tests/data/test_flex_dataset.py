import pytest
import unittest

import numpy as np

from flex.data.flex_dataset import FlexClientDataset
from flex.data.flex_dataset import FlexDataset


class TestFlexClientDataset(unittest.TestCase):
    def test_X_data_property(self):
        X_data = np.random.rand(100).reshape([20,5])
        # y_data = np.random.choice(2, 100)
        fcd = FlexClientDataset(X_data=X_data)
        assert type(X_data) == type(fcd.X_data)
        assert np.array_equal(X_data, fcd.X_data)

    def test_y_data_property(self):
        X_data = np.random.rand(100).reshape([20,5])
        y_data = np.random.choice(2, 20)
        fcd = FlexClientDataset(X_data=X_data, y_data=y_data)
        assert np.array_equal(X_data, fcd.X_data)
        assert np.array_equal(y_data, fcd.y_data)
        assert X_data.shape[0] == y_data.shape[0]

    def test_X_names_property_valid_X_names(self):
        X_data = np.random.rand(100).reshape([20,5])
        y_data = np.random.choice(2, 20)
        fcd = FlexClientDataset(X_data=X_data, y_data=y_data)
        X_names = [f"x{i}" for i in range(X_data.shape[1])]
        assert X_names == fcd.X_names
        assert X_data.shape[0] == y_data.shape[0]

    def test_X_names_property_invalid_X_names(self):
        X_data = np.random.rand(100).reshape([20,5])
        y_data = np.random.choice(2, 20)
        fcd = FlexClientDataset(X_data=X_data, y_data=y_data)
        X_names = [f"x{i}" for i in range(X_data.shape[1]+4)]
        assert X_names != fcd.X_names
        assert X_data.shape[0] == y_data.shape[0]

    def test_y_names_property(self):
        X_data = np.random.rand(100).reshape([20,5])
        y_data = np.random.choice(2, 20)
        fcd = FlexClientDataset(X_data=X_data, y_data=y_data)
        y_names = [f"class_{c}" for c in np.unique(y_data)]
        assert y_names == fcd.y_names
        assert X_data.shape[0] == y_data.shape[0]


class TestFlexDataset(unittest.TestCase):
    def test_get_method(self):
        X_data = np.random.rand(100).reshape([20,5])
        y_data = np.random.choice(2, 20)
        fcd = FlexClientDataset(X_data=X_data, y_data=y_data)
        flex_data = FlexDataset()
        flex_data['client_1'] = fcd
        assert flex_data['client_1'] == fcd
