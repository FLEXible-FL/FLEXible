import unittest

import pytest

from flex.data import FlexDataDistribution, FlexDatasetConfig


@pytest.fixture(name="basic_config")
def fixture_simple_fex_data_object():
    return FlexDatasetConfig(
        seed=0,
        n_clients=2,
        replacement=False,
        client_names=["client_0", "client_1"],
    )


class TestFlexDataDistribution(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_simple_fex_data_object(self, basic_config):
        self._config = basic_config

    def test_from_pytorch_text_dataset(self):
        from torchtext.datasets import AG_NEWS

        data = AG_NEWS(split="train")
        flex_dataset = FlexDataDistribution.from_pytorch_text_dataset(
            data, self._config
        )
        assert len(flex_dataset) == self._config.n_clients
        assert len(flex_dataset["client_0"]) == len(flex_dataset["client_1"])

    def test_from_huggingface_text_dataset(self):
        from datasets import load_dataset

        data = load_dataset("ag_news", split="train")
        X_columns = "text"
        label_column = "label"
        flex_dataset = FlexDataDistribution.from_huggingface_dataset(
            data, self._config, X_columns, label_column
        )
        assert len(flex_dataset) == self._config.n_clients
        assert len(flex_dataset["client_0"]) == len(flex_dataset["client_1"])
        assert (
            len(flex_dataset["client_0"]) + len(flex_dataset["client_1"])
            == data.num_rows
        )