import unittest

import pytest

from flex.model import FlexModel


class TestFlexModel(unittest.TestCase):
    def test_getter_actor_id(self):
        flex_model = FlexModel()
        flex_model.actor_id = "server"
        assert flex_model.actor_id == "server"

    def test_setter_actor_id(self):
        flex_model = FlexModel()
        flex_model.actor_id = "server"
        with pytest.raises(PermissionError):
            flex_model.actor_id = "client_1"

    def test_deleter_actor_id(self):
        flex_model = FlexModel()
        flex_model.actor_id = "server"
        del flex_model.actor_id
        assert flex_model.actor_id == "server"
