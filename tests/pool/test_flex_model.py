import unittest

import pytest

from flex.pool import FlexModel


class TestFlexModel(unittest.TestCase):
    def test__getattr__(self):
        flex_model = FlexModel()
        flex_model["model"] = "test"
        model = flex_model.model
        model2 = flex_model["model"]
        assert model == model2

    def test_setitem__one_use_keys_repeated(self):
        flex_model = FlexModel()
        flex_model["model"] = "test"
        with pytest.raises(KeyError):
            flex_model["model"] = "test2"

    def test__delattr__(self):
        flex_model = FlexModel()
        flex_model["model"] = "test"
        del flex_model["model"]
        assert "model" not in flex_model.keys()

    def test__delattr__key_not_exists(self):
        flex_model = FlexModel()
        flex_model["model"] = "test"
        with pytest.raises(KeyError):
            del flex_model["weights"]

    def test__getattr__error(self):
        flex_model = FlexModel()
        flex_model.model = "test"
        with pytest.raises(KeyError):
            flex_model.weights

    def test_setattr__one_use_keys_repeated(self):
        flex_model = FlexModel()
        flex_model.model = "test"
        with pytest.raises(KeyError):
            flex_model.model = "test2"