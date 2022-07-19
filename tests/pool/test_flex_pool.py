import unittest

import numpy as np
import pytest

from flex.data.flex_dataset import FlexDataObject, FlexDataset
from flex.pool.actors import FlexActors, FlexRole, FlexRoleManager
from flex.pool.flex_pool import FlexPoolManager


@pytest.fixture(name="fld")
def fixture_flex_dataset():
    """Function that returns a FlexDataset provided as example to test functions.

    Returns:
        FlexDataset: A FlexDataset generated randomly
    """
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    fcd = FlexDataObject(X_data=X_data, y_data=y_data)
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    fcd1 = FlexDataObject(X_data=X_data, y_data=y_data)
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    fcd2 = FlexDataObject(X_data=X_data, y_data=y_data)
    return FlexDataset({"client_1": fcd, "client_2": fcd1, "client_3": fcd2})


@pytest.fixture(name="only_clients")
def fixture_only_clients():
    client1_id = "client_1"
    role1 = FlexRole.client
    client2_id = "client_2"
    role2 = FlexRole.client
    client3_id = "client_3"
    role3 = FlexRole.client
    return FlexActors(
        {
            client1_id: role1,
            client2_id: role2,
            client3_id: role3,
        }
    )


class TestRoleManger(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_only_clients(self, only_clients):
        self._only_clients = only_clients

    @pytest.fixture(autouse=True)
    def _fixture_flex_dataset(self, fld):
        self._fld = fld

    def test_client_server_architecture(self):
        p = FlexPoolManager.client_server_architecture(self._fld)
        assert FlexRoleManager.is_client(p._actors["client_1"]) is True
        assert FlexRoleManager.is_aggregator(p._actors["client_1"]) is False
        assert FlexRoleManager.is_server(p._actors["client_1"]) is False

        assert FlexRoleManager.is_client(p._actors["client_2"]) is True
        assert FlexRoleManager.is_aggregator(p._actors["client_2"]) is False
        assert FlexRoleManager.is_server(p._actors["client_2"]) is False

        assert FlexRoleManager.is_client(p._actors["client_3"]) is True
        assert FlexRoleManager.is_aggregator(p._actors["client_3"]) is False
        assert FlexRoleManager.is_server(p._actors["client_3"]) is False

        assert FlexRoleManager.is_client(p._actors["server"]) is False
        assert FlexRoleManager.is_aggregator(p._actors["server"]) is True
        assert FlexRoleManager.is_server(p._actors["server"]) is True

    def test_p2p_architecture(self):
        p = FlexPoolManager.p2p_architecture(self._fld)
        assert FlexRoleManager.is_client(p._actors["client_1"]) is True
        assert FlexRoleManager.is_aggregator(p._actors["client_1"]) is True
        assert FlexRoleManager.is_server(p._actors["client_1"]) is True

        assert FlexRoleManager.is_client(p._actors["client_2"]) is True
        assert FlexRoleManager.is_aggregator(p._actors["client_2"]) is True
        assert FlexRoleManager.is_server(p._actors["client_2"]) is True

        assert FlexRoleManager.is_client(p._actors["client_3"]) is True
        assert FlexRoleManager.is_aggregator(p._actors["client_3"]) is True
        assert FlexRoleManager.is_server(p._actors["client_3"]) is True

    def test_validate_no_server_no_aggregator(self):
        with pytest.raises(ValueError):
            FlexPoolManager(self._fld, self._only_clients)

    def test_validate_client_without_data(self):
        fld = FlexDataset(
            {"client_1": self._fld["client_1"], "client_2": self._fld["client_2"]}
        )
        with pytest.raises(ValueError):
            FlexPoolManager(fld, self._only_clients)

    def test_filer(self):
        p = FlexPoolManager.p2p_architecture(self._fld)
        p.filter(lambda a: a)
        pass
