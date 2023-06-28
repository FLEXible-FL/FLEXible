import unittest
from collections import defaultdict

import numpy as np
import pytest
from sklearn.datasets import load_iris

from flex.actors.actors import FlexActors, FlexRole, FlexRoleManager
from flex.data import FedDataDistribution
from flex.data.fed_dataset import Dataset, FedDataset
from flex.model.model import FlexModel
from flex.pool.pool import FlexPool


@pytest.fixture(name="fld")
def fixture_flex_dataset():
    """Function that returns a FlexDataset provided as example to test functions.

    Returns:
        FedDataset: A FlexDataset generated randomly
    """
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    fcd = Dataset(X_data=X_data, y_data=y_data)
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    fcd1 = Dataset(X_data=X_data, y_data=y_data)
    X_data = np.random.rand(100).reshape([20, 5])
    y_data = np.random.choice(2, 20)
    fcd2 = Dataset(X_data=X_data, y_data=y_data)
    return FedDataset({"client_1": fcd, "client_2": fcd1, "client_3": fcd2})


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


class TestFlexPool(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_only_clients(self, only_clients):
        self._only_clients = only_clients

    @pytest.fixture(autouse=True)
    def _fixture_iris_dataset(self):
        iris = load_iris()
        tmp = Dataset(X_data=iris.data, y_data=iris.target)
        self._iris = FedDataDistribution.iid_distribution(tmp, n_clients=2)

    @pytest.fixture(autouse=True)
    def _fixture_flex_dataset(self, fld):
        self._fld = fld

    def test_len_property(self):
        p = FlexPool.client_server_architecture(self._iris, lambda *args: None)
        assert len(p) != len(self._iris)
        assert len(p.filter(lambda *args: True)) == len(p)
        assert len(p.actor_ids) == len(p)

    def test_check_compatibility(self):
        p = FlexPool.client_server_architecture(self._fld, lambda *args: None)
        server_pool = p.servers
        client_pool = p.clients
        assert FlexPool.check_compatibility(server_pool, client_pool) is True
        assert (
            FlexPool.check_compatibility(client_pool, server_pool) is True
        )  # Servers are also aggregators in this architecture
        assert FlexPool.check_compatibility(client_pool, client_pool) is False

    def test_map(self):
        p = FlexPool.client_server_architecture(self._fld, lambda *args: None)
        server_pool = p.servers
        client_pool = p.clients
        assert server_pool.map(lambda *args: True, client_pool) == [True]
        assert client_pool.map(lambda *args: True) == len(client_pool) * [True]
        with pytest.raises(ValueError):
            assert client_pool.map(lambda *args: True, client_pool)

    def test_client_server_architecture(self):
        p = FlexPool.client_server_architecture(self._fld, lambda *args: None)
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
        p = FlexPool.p2p_architecture(self._fld, lambda *args: None)
        assert FlexRoleManager.is_client(p._actors["client_1"]) is True
        assert FlexRoleManager.is_aggregator(p._actors["client_1"]) is True
        assert FlexRoleManager.is_server(p._actors["client_1"]) is True

        assert FlexRoleManager.is_client(p._actors["client_2"]) is True
        assert FlexRoleManager.is_aggregator(p._actors["client_2"]) is True
        assert FlexRoleManager.is_server(p._actors["client_2"]) is True

        assert FlexRoleManager.is_client(p._actors["client_3"]) is True
        assert FlexRoleManager.is_aggregator(p._actors["client_3"]) is True
        assert FlexRoleManager.is_server(p._actors["client_3"]) is True

    def test_validate_client_without_data(self):
        fld = FedDataset(
            {"client_1": self._fld["client_1"], "client_2": self._fld["client_2"]}
        )
        with pytest.raises(ValueError):
            FlexPool(fld, self._only_clients, None)

    def test_validate_client_without_all_models(self):
        models = defaultdict(None, {"client_1": 0, "client_2": 0})
        with pytest.raises(ValueError):
            FlexPool(self._fld, self._only_clients, models)

    def test_validate_client_without_role(self):
        client1_id = "client_1"
        role1 = FlexRole.client
        client2_id = "client_2"
        role2 = FlexRole.client
        missing_one = FlexActors(
            {
                client1_id: role1,
                client2_id: role2,
            }
        )
        with pytest.raises(ValueError):
            FlexPool(self._fld, missing_one, defaultdict())

    def test_validate_client_without_model(self):
        p = FlexPool.p2p_architecture(self._fld, init_func=lambda *args: None)
        assert p._models.get("client_1") == FlexModel()
        assert p._models.get("client_2") == FlexModel()
        assert p._models.get("client_3") == FlexModel()

    def test_filter(self):
        p = FlexPool.p2p_architecture(self._fld, init_func=lambda *args: None)
        new_p = p.filter(lambda a, b: FlexRoleManager.is_client(b))
        assert all(
            FlexRoleManager.is_client(actor_role)
            for _, actor_role in new_p._actors.items()
        )

    def test_filter_dropout(self):
        iris = load_iris()
        tmp = Dataset(X_data=iris.data, y_data=iris.target)
        self._iris_many_clients = FedDataDistribution.iid_distribution(
            tmp, n_clients=100
        )
        p = FlexPool.client_server_architecture(self._iris, lambda *args: None)
        pool_size = len(p)
        assert all(
            len(p.filter(node_dropout=1 - (i / pool_size))) == i
            for i in range(pool_size)
        )

    def test_client_property(self):
        p = FlexPool.p2p_architecture(self._fld, init_func=lambda *args: None)
        new_p = p.clients
        assert all(
            FlexRoleManager.is_client(actor_role)
            for _, actor_role in new_p._actors.items()
        )

    def test_aggregator_property(self):
        p = FlexPool.p2p_architecture(self._fld, init_func=lambda *args: None)
        new_p = p.aggregators
        assert all(
            FlexRoleManager.is_aggregator(actor_role)
            for _, actor_role in new_p._actors.items()
        )

    def test_server_property(self):
        p = FlexPool.p2p_architecture(self._fld, init_func=lambda *args: None)
        new_p = p.servers
        assert all(
            FlexRoleManager.is_server(actor_role)
            for _, actor_role in new_p._actors.items()
        )

    def test_filter_dropout_correct(self):
        p = FlexPool.p2p_architecture(self._fld, init_func=lambda *args: None)
        clients = p.filter(lambda a, b: FlexRoleManager.is_client(b), node_dropout=0.33)
        assert len(clients._actors) == 2

    def test_filter_dropout_greater_one(self):
        p = FlexPool.p2p_architecture(self._fld, init_func=lambda *args: None)
        clients = p.filter(lambda a, b: FlexRoleManager.is_client(b), node_dropout=10)
        assert len(clients._actors) == 0

    def test_reserved_server_id(self):
        fld = FedDataset(
            {"server": self._fld["client_1"], "client_2": self._fld["client_2"]}
        )
        with pytest.raises(ValueError):
            FlexPool.client_server_architecture(fld, init_func=lambda *args: None)
