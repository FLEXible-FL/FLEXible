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


class TestFlexDataObject(unittest.TestCase):
    def test_validate_from_torchtext_dataset(self):
        from torchtext.datasets import AG_NEWS

        data = AG_NEWS(split="train")
        fcd = Dataset.from_torchtext_dataset(data)
        fcd.validate()

    def test_validate_from_torchtext_dataset_raise_error(self):
        data = np.random.rand(100).reshape([20, 5])
        with pytest.raises(ValueError):
            Dataset.from_torchtext_dataset(data)

    def test_validate_from_huggingface_dataset(self):
        from datasets import load_dataset

        data = load_dataset("ag_news", split="train")
        X_columns = "text"
        label_column = "label"
        fcd = Dataset.from_huggingface_datasets(
            data, X_columns=X_columns, label_column=label_column
        )
        fcd.validate()

    def test_validate_from_huggingface_dataset_error(self):
        data = np.random.rand(100).reshape([20, 5])
        X_columns = "text"
        label_column = "label"
        with pytest.raises(ValueError):
            Dataset.from_huggingface_datasets(data, X_columns, label_column)
