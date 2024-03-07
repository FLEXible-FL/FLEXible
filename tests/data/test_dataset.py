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
import unittest

import numpy as np
import pytest

from flex.data.dataset import Dataset


@pytest.fixture(name="fcd")
def fixture_simple_fex_data_object():
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    return Dataset.from_array(X_data, y_data)


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
        fcd = Dataset.from_array(X_data, y_data)
        for x, y, (x_bis, y_bis) in zip(X_data, y_data, fcd):
            assert np.array_equal(x, x_bis)
            assert y == y_bis

    def test_getitem_property_y_data_none(self):
        X_data = np.random.rand(100).reshape([20, 5])
        fcd = Dataset.from_array(X_data)
        for x, (x_bis, y_bis) in zip(X_data, fcd):
            assert np.array_equal(x, x_bis)
            assert y_bis is None

    def test_validate_correct_object(self):
        self._fcd.validate()

    def test_len_X_data_differs_len_y_data(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 19)
        fcd = Dataset.from_array(X_data, y_data)
        with pytest.raises(ValueError):
            fcd.validate()

    def test_validate_from_torchtext_dataset(self):
        from torchtext.datasets import AG_NEWS

        data = AG_NEWS(split="train")
        fcd = Dataset.from_torchtext_dataset(data)
        fcd.validate()

    def test_validate_from_huggingface_dataset(self):
        from datasets import load_dataset

        data = load_dataset("ag_news", split="train")
        X_columns = ["text"]
        label_columns = ["label"]
        fcd = Dataset.from_huggingface_dataset(
            data, X_columns=X_columns, label_columns=label_columns
        )
        fcd.validate()

    def test_validate_from_huggingface_dataset_lazy(self):
        from datasets import load_dataset

        data = load_dataset("ag_news", split="train")
        X_columns = ["text"]
        label_columns = ["label"]
        fcd = Dataset.from_huggingface_dataset(
            data, X_columns=X_columns, label_columns=label_columns
        )
        fcd.validate()

    def test_validate_from_huggingface_dataset_lazy_with_str_no_subset(self):
        data = "ag_news;train"
        X_columns = ["text"]
        label_columns = ["label"]
        fcd = Dataset.from_huggingface_dataset(
            data, X_columns=X_columns, label_columns=label_columns
        )
        fcd.validate()

    def test_validate_from_huggingface_dataset_lazy_with_str_and_subset(self):
        data = "tweet_eval;emoji;train"
        X_columns = ["text"]
        label_columns = ["label"]
        fcd = Dataset.from_huggingface_dataset(
            data, X_columns=X_columns, label_columns=label_columns
        )
        fcd.validate()

    def test_to_torchvision_dataset_w_flex_datasets(self):
        import torch
        from torchvision import transforms

        from flex.datasets import load

        fcd, _ = load("emnist", split="digits")
        torch_fcd = fcd.to_torchvision_dataset(
            transform=transforms.ToTensor(),
            target_transform=transforms.Compose(
                [
                    lambda x: torch.as_tensor(x).long(),
                    lambda x: torch.nn.functional.one_hot(x, 10),
                ]
            ),
        )
        batch_size = 64
        dataloader = torch.utils.data.DataLoader(
            torch_fcd, batch_size=batch_size, shuffle=True
        )
        train_features, train_labels = next(iter(dataloader))

        assert len(train_features) == batch_size
        assert torch.is_tensor(train_features)
        assert len(train_labels) == batch_size
        assert torch.is_tensor(train_labels)
        assert len(train_labels[0]) == 10  # number of labels

    def test_to_tf_dataset_w_flex_datasets(self):
        import tensorflow as tf

        from flex.datasets import load

        fcd, _ = load("emnist", split="digits")
        tf_fcd = fcd.to_tf_dataset()
        batch_size = 64
        tf_fcd = tf_fcd.batch(batch_size)
        train_features, train_labels = next(iter(tf_fcd))

        assert len(train_features) == batch_size
        assert tf.is_tensor(train_features)
        assert len(train_labels) == batch_size
        assert tf.is_tensor(train_labels)

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

    def test_from_array_arrays(self):
        X_data = np.random.rand(100).reshape([20, 5])
        y_data = np.random.choice(2, 20)
        fcd = Dataset.from_array(X_data, y_data)
        assert len(fcd.X_data) == len(self._fcd.X_data)
        assert len(fcd.y_data) == len(self._fcd.y_data)

    def test_from_array_arrays_y_none(self):
        X_data = np.random.rand(100).reshape([20, 5])
        fcd = Dataset.from_array(X_data)
        assert len(fcd.X_data) == len(self._fcd.X_data)

    def test_from_arrays(self):
        X_data = list(np.random.rand(100).reshape([20, 5]))
        y_data = list(np.random.choice(2, 20))
        fcd = Dataset.from_array(X_data, y_data)
        assert len(fcd.X_data) == len(self._fcd.X_data)
        assert len(fcd.y_data) == len(self._fcd.y_data)

    def test_from_array_y_none(self):
        X_data = list(np.random.rand(100).reshape([20, 5]))
        fcd = Dataset.from_array(X_data)
        assert len(fcd.X_data) == len(self._fcd.X_data)

    def test_to_list(self):
        X_data, y_data = self._fcd.to_list()
        assert isinstance(X_data, list)
        assert isinstance(y_data, list)
        assert len(X_data) == len(self._fcd.X_data)
        assert len(y_data) == len(self._fcd.y_data)

    def test_to_list_y_none(self):
        X_data = np.random.rand(100).reshape([20, 5])
        fcd = Dataset.from_array(X_data)
        X_data = fcd.to_list()
        assert isinstance(X_data, list)
        assert len(X_data) == len(self._fcd.X_data)

    def test_to_numpy(self):
        X_data, y_data = self._fcd.to_numpy(x_dtype=np.int16, y_dtype=np.int16)
        assert isinstance(X_data, np.ndarray)
        assert isinstance(y_data, np.ndarray)
        assert len(X_data) == len(self._fcd.X_data)
        assert len(y_data) == len(self._fcd.y_data)

    def test_to_numpy_no_dtype(self):
        X_data, y_data = self._fcd.to_numpy()
        assert isinstance(X_data, np.ndarray)
        assert isinstance(y_data, np.ndarray)
        assert len(X_data) == len(self._fcd.X_data)
        assert len(y_data) == len(self._fcd.y_data)

    def test_to_numpy_y_none(self):
        X_data = np.random.rand(100).reshape([20, 5])
        fcd = Dataset.from_array(X_data)
        X_data = fcd.to_numpy(x_dtype=np.int16)
        assert isinstance(X_data, np.ndarray)
        assert len(X_data) == len(self._fcd.X_data)
