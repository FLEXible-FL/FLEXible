import unittest

import pytest

from flex.actors.actors import FlexActors
from flex.actors.role import FlexRole
from flex.actors.role_manager import FlexRoleManager
from flex.actors.architectures import client_server_architecture, p2p_architecture


@pytest.fixture(name="nodes_ids")
def fixture_nodes_ids():
    """Function that returns a Iterable provided as example to test functions.

    Returns:
        List: A list with the node ids.
    """
    return ['client_1', 'client_2', 'client_3']


class TestArchitectures(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_nodes_ids(self, nodes_ids):
        self._nodes_ids = nodes_ids

    def test_client_server_architecture(self):
        architecture = client_server_architecture(self._nodes_ids)
        assert FlexRoleManager.is_client(architecture["client_1"]) is True
        assert FlexRoleManager.is_aggregator(architecture["client_1"]) is False
        assert FlexRoleManager.is_server(architecture["client_1"]) is False

        assert FlexRoleManager.is_client(architecture["client_2"]) is True
        assert FlexRoleManager.is_aggregator(architecture["client_2"]) is False
        assert FlexRoleManager.is_server(architecture["client_2"]) is False

        assert FlexRoleManager.is_client(architecture["client_3"]) is True
        assert FlexRoleManager.is_aggregator(architecture["client_3"]) is False
        assert FlexRoleManager.is_server(architecture["client_3"]) is False

        assert FlexRoleManager.is_client(architecture["server"]) is False
        assert FlexRoleManager.is_aggregator(architecture["server"]) is True
        assert FlexRoleManager.is_server(architecture["server"]) is True

    def test_p2p_architecture(self):
        architecture = p2p_architecture(self._nodes_ids)
        assert FlexRoleManager.is_client(architecture["client_1"]) is True
        assert FlexRoleManager.is_aggregator(architecture["client_1"]) is True
        assert FlexRoleManager.is_server(architecture["client_1"]) is True

        assert FlexRoleManager.is_client(architecture["client_2"]) is True
        assert FlexRoleManager.is_aggregator(architecture["client_2"]) is True
        assert FlexRoleManager.is_server(architecture["client_2"]) is True

        assert FlexRoleManager.is_client(architecture["client_3"]) is True
        assert FlexRoleManager.is_aggregator(architecture["client_3"]) is True
        assert FlexRoleManager.is_server(architecture["client_3"]) is True
