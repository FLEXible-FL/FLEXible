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

    def test_validate_from_torchtext_dataset(self):
        from torchtext.datasets import AG_NEWS

        data = AG_NEWS(split="train")
        fcd = FlexDataObject.from_torchtext_dataset(data)
        fcd.validate()

    def test_validate_from_huggingface_dataset(self):
        from datasets import load_dataset

        data = load_dataset("ag_news", split="train")
        X_columns = "text"
        label_column = "label"
        fcd = FlexDataObject.from_huggingface_dataset(
            data, X_columns=X_columns, label_column=label_column
        )
        fcd.validate()

    def test_pluggable_datasets_in_property(self):
        from torchtext.datasets import AG_NEWS
        from torchvision.datasets import MNIST

        from flex.data.pluggable_datasets import (
            PluggableTorchtext,
            PluggableTorchvision,
        )

        assert AG_NEWS.__name__ in PluggableTorchtext
        assert "random_string" not in PluggableTorchtext
        assert MNIST.__name__ in PluggableTorchvision
        assert "random_string" not in PluggableTorchvision
