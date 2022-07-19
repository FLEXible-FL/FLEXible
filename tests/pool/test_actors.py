import unittest

import pytest

from flex.pool.actors import Actors, Role, RoleManager


@pytest.fixture(name="actors_cs")
def fixture_actors():
    client1_id = "client1"
    role1 = Role.client
    client2_id = "client2"
    role2 = Role.client
    client3_id = "client3"
    role3 = Role.client
    aggregator_id = "aggregator"
    role4 = Role.aggregator
    server_id = "server"
    role5 = Role.server
    return Actors(
        {
            client1_id: role1,
            client2_id: role2,
            client3_id: role3,
            aggregator_id: role4,
            server_id: role5,
        }
    )


@pytest.fixture(name="p2p")
def fixture_peer_to_peer():
    actor1_id = "actor1"
    role1 = Role.server_aggregator_client
    actor2_id = "actor2"
    role2 = Role.server_aggregator_client
    actor3_id = "actor3"
    role3 = Role.server_aggregator_client
    actor4_id = "actor4"
    role4 = Role.server_aggregator_client
    actor5_id = "actor5"
    role5 = Role.server_aggregator_client
    return Actors(
        {
            actor1_id: role1,
            actor2_id: role2,
            actor3_id: role3,
            actor4_id: role4,
            actor5_id: role5,
        }
    )


class TestRoleManger(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_simple_actors_client_server(self, actors_cs):
        self._actors_cs = actors_cs

    @pytest.fixture(autouse=True)
    def _fixture_peer_to_peer(self, p2p):
        self._p2p = p2p

    def test_is_client_method(self):
        # Client-Server Architecture
        assert RoleManager.isClient(self._actors_cs["client1"]) is True
        assert RoleManager.isClient(self._actors_cs["client2"]) is True
        assert RoleManager.isClient(self._actors_cs["client3"]) is True
        assert RoleManager.isClient(self._actors_cs["aggregator"]) is False
        assert RoleManager.isClient(self._actors_cs["server"]) is False
        # Peer-To-Peer Architecture
        assert RoleManager.isClient(self._p2p["actor1"]) is True
        assert RoleManager.isClient(self._p2p["actor2"]) is True
        assert RoleManager.isClient(self._p2p["actor3"]) is True
        assert RoleManager.isClient(self._p2p["actor4"]) is True
        assert RoleManager.isClient(self._p2p["actor5"]) is True

    def test_is_aggregator_method(self):
        # Client-Server Architecture
        assert RoleManager.isAggregator(self._actors_cs["client1"]) is False
        assert RoleManager.isAggregator(self._actors_cs["client2"]) is False
        assert RoleManager.isAggregator(self._actors_cs["client3"]) is False
        assert RoleManager.isAggregator(self._actors_cs["aggregator"]) is True
        assert RoleManager.isAggregator(self._actors_cs["server"]) is False
        # Peer-To-Peer Architecture
        assert RoleManager.isAggregator(self._p2p["actor1"]) is True
        assert RoleManager.isAggregator(self._p2p["actor2"]) is True
        assert RoleManager.isAggregator(self._p2p["actor3"]) is True
        assert RoleManager.isAggregator(self._p2p["actor4"]) is True
        assert RoleManager.isAggregator(self._p2p["actor5"]) is True

    def test_is_server_method(self):
        # Client-Server Architecture
        assert RoleManager.isServer(self._actors_cs["client1"]) is False
        assert RoleManager.isServer(self._actors_cs["client2"]) is False
        assert RoleManager.isServer(self._actors_cs["client3"]) is False
        assert RoleManager.isServer(self._actors_cs["aggregator"]) is False
        assert RoleManager.isServer(self._actors_cs["server"]) is True
        # Peer-To-Peer Architecture
        assert RoleManager.isServer(self._p2p["actor1"]) is True
        assert RoleManager.isServer(self._p2p["actor2"]) is True
        assert RoleManager.isServer(self._p2p["actor3"]) is True
        assert RoleManager.isServer(self._p2p["actor4"]) is True
        assert RoleManager.isServer(self._p2p["actor5"]) is True

    def test_check_compatibility_method(self):
        # Client-Server Architecture
        assert (
            RoleManager.check_compatibility(
                self._actors_cs["client1"], self._actors_cs["server"]
            )
            is False
        )
        assert (
            RoleManager.check_compatibility(
                self._actors_cs["server"], self._actors_cs["client1"]
            )
            is True
        )
        assert (
            RoleManager.check_compatibility(
                self._actors_cs["server"], self._actors_cs["aggregator"]
            )
            is False
        )
        assert (
            RoleManager.check_compatibility(
                self._actors_cs["client1"], self._actors_cs["client2"]
            )
            is False
        )
        assert (
            RoleManager.check_compatibility(
                self._actors_cs["aggregator"], self._actors_cs["server"]
            )
            is True
        )
        assert (
            RoleManager.check_compatibility(
                self._actors_cs["client1"], self._actors_cs["aggregator"]
            )
            is True
        )
        # Peer-To-Peer Architecture
        assert (
            RoleManager.check_compatibility(self._p2p["actor1"], self._p2p["actor2"])
            is True
        )
        assert (
            RoleManager.check_compatibility(self._p2p["actor1"], self._p2p["actor3"])
            is True
        )
        assert (
            RoleManager.check_compatibility(self._p2p["actor1"], self._p2p["actor4"])
            is True
        )
        assert (
            RoleManager.check_compatibility(self._p2p["actor1"], self._p2p["actor5"])
            is True
        )
        assert (
            RoleManager.check_compatibility(self._p2p["actor2"], self._p2p["actor3"])
            is True
        )
        assert (
            RoleManager.check_compatibility(self._p2p["actor2"], self._p2p["actor4"])
            is True
        )
        assert (
            RoleManager.check_compatibility(self._p2p["actor2"], self._p2p["actor5"])
            is True
        )
        assert (
            RoleManager.check_compatibility(self._p2p["actor3"], self._p2p["actor4"])
            is True
        )
        assert (
            RoleManager.check_compatibility(self._p2p["actor3"], self._p2p["actor5"])
            is True
        )
        assert (
            RoleManager.check_compatibility(self._p2p["actor4"], self._p2p["actor5"])
            is True
        )
        assert (
            RoleManager.check_compatibility(self._p2p["actor5"], self._p2p["actor3"])
            is True
        )


class TestActors(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_simple_actors_client_server(self, actors_cs):
        self._actors_cs = actors_cs

    @pytest.fixture(autouse=True)
    def _fixture_peer_to_peer(self, p2p):
        self._p2p = p2p

    def test_check_compatibility_method(self):
        # Client-Server Architecture
        assert self._actors_cs.check_compatibility("client1", "client2") is False
        assert self._actors_cs.check_compatibility("client1", "server") is False
        assert self._actors_cs.check_compatibility("client1", "aggregator") is True
        assert self._actors_cs.check_compatibility("aggregator", "client3") is False
        assert self._actors_cs.check_compatibility("aggregator", "server") is True
        assert self._actors_cs.check_compatibility("server", "aggregator") is False
        # Peer-To-Peer Architecture
        assert self._p2p.check_compatibility("actor1", "actor2") is True
        assert self._p2p.check_compatibility("actor1", "actor3") is True
        assert self._p2p.check_compatibility("actor1", "actor4") is True
        assert self._p2p.check_compatibility("actor1", "actor5") is True
        assert self._p2p.check_compatibility("actor2", "actor3") is True
        assert self._p2p.check_compatibility("actor2", "actor4") is True
        assert self._p2p.check_compatibility("actor2", "actor5") is True
        assert self._p2p.check_compatibility("actor3", "actor4") is True
        assert self._p2p.check_compatibility("actor3", "actor5") is True
        assert self._p2p.check_compatibility("actor4", "actor5") is True
        assert self._p2p.check_compatibility("actor5", "actor3") is True

    def test_set_function(self):
        client4_id = "client4"
        role_client4 = Role.client
        self._actors_cs[client4_id] = role_client4
        assert client4_id in self._actors_cs.keys()
        assert self._actors_cs[client4_id] == role_client4
