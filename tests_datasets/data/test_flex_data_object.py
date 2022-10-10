import unittest

import numpy as np
import pytest

from flex.data.flex_data_object import FlexDataObject


class TestFlexDataObject(unittest.TestCase):
    def test_validate_from_torchtext_dataset(self):
        from torchtext.datasets import AG_NEWS

        data = AG_NEWS(split="train")
        fcd = FlexDataObject.from_torchtext_dataset(data)
        fcd.validate()

    def test_validate_from_torchtext_dataset_raise_error(self):
        data = np.random.rand(100).reshape([20, 5])
        with pytest.raises(ValueError):
            FlexDataObject.from_torchtext_dataset(data)

    def test_validate_from_huggingface_dataset(self):
        from datasets import load_dataset

        data = load_dataset("ag_news", split="train")
        X_columns = "text"
        label_column = "label"
        fcd = FlexDataObject.from_huggingface_dataset(
            data, X_columns=X_columns, label_column=label_column
        )
        fcd.validate()

    def test_validate_from_huggingface_dataset_error(self):
        data = np.random.rand(100).reshape([20, 5])
        X_columns = "text"
        label_column = "label"
        with pytest.raises(ValueError):
            FlexDataObject.from_huggingface_dataset(data, X_columns, label_column)
