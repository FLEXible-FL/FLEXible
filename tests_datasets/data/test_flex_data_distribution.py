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

import pytest

from flex.data import FedDataDistribution, FedDatasetConfig


@pytest.fixture(name="basic_config")
def fixture_simple_fex_data_object():
    return FedDatasetConfig(
        seed=0,
        n_nodes=2,
        replacement=False,
        node_ids=["client_0", "client_1"],
    )


class TestFlexDataDistribution(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_simple_fex_data_object(self, basic_config):
        self._config = basic_config

    def test_from_pytorch_text_dataset(self):
        from torchtext.datasets import AG_NEWS

        data = AG_NEWS(split="train")
        flex_dataset = FedDataDistribution.from_config_with_torchtext_dataset(
            data, self._config
        )
        assert len(flex_dataset) == self._config.n_nodes
        assert len(flex_dataset["client_0"]) == len(flex_dataset["client_1"])

    def test_from_huggingface_text_dataset(self):
        from datasets import load_dataset

        data = load_dataset("ag_news", split="train")
        X_columns = "text"
        label_column = "label"
        flex_dataset = FedDataDistribution.from_config_with_huggingface_dataset(
            data, self._config, X_columns, label_column, lazy=False
        )
        assert len(flex_dataset) == self._config.n_nodes
        assert len(flex_dataset["client_0"]) == len(flex_dataset["client_1"])
        assert (
            len(flex_dataset["client_0"]) + len(flex_dataset["client_1"])
            == data.num_rows
        )
