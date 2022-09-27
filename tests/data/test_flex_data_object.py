import unittest

import numpy as np
import pytest

from flex.data.flex_data_object import FlexDataObject


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
        with pytest.raises(AttributeError):
            self._fcd.X_data = X_data

    def test_y_data_property(self):
        y_data = np.random.choice(2, 20)
        with pytest.raises(AttributeError):
            self._fcd.y_data = y_data

    def test_len_property(self):
        X_data = np.random.rand(100).reshape([20, 5])
        assert len(self._fcd) == len(X_data)

    def test_getitem_property(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        for x, y, (x_bis, y_bis) in zip(X_data, y_data, fcd):
            assert np.array_equal(x, x_bis)
            assert y == y_bis

    def test_getitem_property_y_data_none(self):
        X_data = np.random.rand(100).reshape([20, 5])
        fcd = FlexDataObject(X_data=X_data, y_data=None)
        assert len(fcd[:, 0]) == fcd.X_data.shape[0]

    def test_validate_correct_object(self):
        self._fcd.validate()

    def test_len_X_data_differs_len_y_data(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 19)
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        with pytest.raises(ValueError):
            fcd.validate()

    def test_validate_incorrect_object(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, [20, 5])
        fcd = FlexDataObject(X_data=X_data, y_data=y_data)
        with pytest.raises(ValueError):
            fcd.validate()
