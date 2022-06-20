import unittest

import numpy as np
import pytest

from flex.data.flex_dataset import FederatedFlexDataObject, FlexDataObject


class TestFlexDataObject(unittest.TestCase):
    def test_X_data_property(self):
        X_data = np.random.rand(100).reshape([20, 5])
        fcd = FlexDataObject(X_data=X_data)
        assert np.array_equal(X_data, fcd.X_data)

    def test_y_data_property(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        assert np.array_equal(y_data, fcd.y_data)
        assert np.array_equal(y_data, fcd.y_data)

    def test_X_names_property_valid_X_names(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        X_names = np.array([f"x{i}" for i in range(X_data.shape[1])])
        fcd = FlexDataObject(X_data=X_data, y_data=y_data, X_names=X_names)
        assert np.array_equal(X_names, fcd.X_names)
        assert X_data.shape[0] == y_data.shape[0]

    def test_X_names_property_invalid_X_names(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        X_names = [f"x{i}" for i in range(X_data.shape[1] + 4)]
        with pytest.raises(Exception):
            FlexDataObject(X_data=X_data, y_data=y_data, X_names=X_names)

    def test_X_names_property_setter(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        X_names = [f"x{i}" for i in range(X_data.shape[1])]
        fcd.setX(X_data, X_names)
        assert np.array_equal(X_names, fcd.X_names)

    def test_y_names_property_valid_y_names(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        y_names = [f"class_{c}" for c in np.unique(y_data)]
        fcd = FlexDataObject(X_data=X_data, y_data=y_data, y_names=y_names)
        assert np.array_equal(y_data, fcd.y_data)
        assert np.array_equal(y_names, fcd.y_names)

    def test_empty_y_names_y_data(self):
        X_data = np.random.rand(100).reshape([20, 5])
        fcd = FlexDataObject(X_data=X_data)
        assert fcd.y_names is None

    def test_y_names_property_setter(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        y_names = [f"class_{c}" for c in np.unique(y_data)]
        fcd.setY(y_data, y_names)
        assert np.array_equal(y_names, fcd.y_names)

    def test_len_property(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        assert len(fcd) == len(X_data)

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


class TestFederatedFlexDataObject(unittest.TestCase):
    def test_get_method(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        flex_data = FederatedFlexDataObject()
        flex_data["client_1"] = fcd
        assert flex_data["client_1"] == fcd
        assert flex_data.get("client_2") is None
