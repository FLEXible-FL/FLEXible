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

from flex.actors.actors import FlexActors
from flex.actors.role import FlexRole
from flex.actors.role_manager import FlexRoleManager


@pytest.fixture(name="actors_cs")
def fixture_actors():
    client1_id = "client1"
    role1 = FlexRole.client
    client2_id = "client2"
    role2 = FlexRole.client
    client3_id = "client3"
    role3 = FlexRole.client
    aggregator_id = "aggregator"
    role4 = FlexRole.aggregator
    server_id = "server"
    role5 = FlexRole.server
    return FlexActors(
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
    role1 = FlexRole.server_aggregator_client
    actor2_id = "actor2"
    role2 = FlexRole.server_aggregator_client
    actor3_id = "actor3"
    role3 = FlexRole.server_aggregator_client
    actor4_id = "actor4"
    role4 = FlexRole.server_aggregator_client
    actor5_id = "actor5"
    role5 = FlexRole.server_aggregator_client
    return FlexActors(
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
        assert FlexRoleManager.is_client(self._actors_cs["client1"]) is True
        assert FlexRoleManager.is_client(self._actors_cs["client2"]) is True
        assert FlexRoleManager.is_client(self._actors_cs["client3"]) is True
        assert FlexRoleManager.is_client(self._actors_cs["aggregator"]) is False
        assert FlexRoleManager.is_client(self._actors_cs["server"]) is False
        # Peer-To-Peer Architecture
        assert FlexRoleManager.is_client(self._p2p["actor1"]) is True
        assert FlexRoleManager.is_client(self._p2p["actor2"]) is True
        assert FlexRoleManager.is_client(self._p2p["actor3"]) is True
        assert FlexRoleManager.is_client(self._p2p["actor4"]) is True
        assert FlexRoleManager.is_client(self._p2p["actor5"]) is True

    def test_is_aggregator_method(self):
        # Client-Server Architecture
        assert FlexRoleManager.is_aggregator(self._actors_cs["client1"]) is False
        assert FlexRoleManager.is_aggregator(self._actors_cs["client2"]) is False
        assert FlexRoleManager.is_aggregator(self._actors_cs["client3"]) is False
        assert FlexRoleManager.is_aggregator(self._actors_cs["aggregator"]) is True
        assert FlexRoleManager.is_aggregator(self._actors_cs["server"]) is False
        # Peer-To-Peer Architecture
        assert FlexRoleManager.is_aggregator(self._p2p["actor1"]) is True
        assert FlexRoleManager.is_aggregator(self._p2p["actor2"]) is True
        assert FlexRoleManager.is_aggregator(self._p2p["actor3"]) is True
        assert FlexRoleManager.is_aggregator(self._p2p["actor4"]) is True
        assert FlexRoleManager.is_aggregator(self._p2p["actor5"]) is True

    def test_is_server_method(self):
        # Client-Server Architecture
        assert FlexRoleManager.is_server(self._actors_cs["client1"]) is False
        assert FlexRoleManager.is_server(self._actors_cs["client2"]) is False
        assert FlexRoleManager.is_server(self._actors_cs["client3"]) is False
        assert FlexRoleManager.is_server(self._actors_cs["aggregator"]) is False
        assert FlexRoleManager.is_server(self._actors_cs["server"]) is True
        # Peer-To-Peer Architecture
        assert FlexRoleManager.is_server(self._p2p["actor1"]) is True
        assert FlexRoleManager.is_server(self._p2p["actor2"]) is True
        assert FlexRoleManager.is_server(self._p2p["actor3"]) is True
        assert FlexRoleManager.is_server(self._p2p["actor4"]) is True
        assert FlexRoleManager.is_server(self._p2p["actor5"]) is True

    def test_check_compatibility_method(self):
        # Client-Server Architecture
        assert (
            FlexRoleManager.check_compatibility(
                self._actors_cs["client1"], self._actors_cs["server"]
            )
            is False
        )
        assert (
            FlexRoleManager.check_compatibility(
                self._actors_cs["server"], self._actors_cs["client1"]
            )
            is True
        )
        assert (
            FlexRoleManager.check_compatibility(
                self._actors_cs["server"], self._actors_cs["aggregator"]
            )
            is False
        )
        assert (
            FlexRoleManager.check_compatibility(
                self._actors_cs["client1"], self._actors_cs["client2"]
            )
            is False
        )
        assert (
            FlexRoleManager.check_compatibility(
                self._actors_cs["aggregator"], self._actors_cs["server"]
            )
            is True
        )
        assert (
            FlexRoleManager.check_compatibility(
                self._actors_cs["client1"], self._actors_cs["aggregator"]
            )
            is True
        )
        # Peer-To-Peer Architecture
        assert (
            FlexRoleManager.check_compatibility(
                self._p2p["actor1"], self._p2p["actor2"]
            )
            is True
        )
        assert (
            FlexRoleManager.check_compatibility(
                self._p2p["actor1"], self._p2p["actor3"]
            )
            is True
        )
        assert (
            FlexRoleManager.check_compatibility(
                self._p2p["actor1"], self._p2p["actor4"]
            )
            is True
        )
        assert (
            FlexRoleManager.check_compatibility(
                self._p2p["actor1"], self._p2p["actor5"]
            )
            is True
        )
        assert (
            FlexRoleManager.check_compatibility(
                self._p2p["actor2"], self._p2p["actor3"]
            )
            is True
        )
        assert (
            FlexRoleManager.check_compatibility(
                self._p2p["actor2"], self._p2p["actor4"]
            )
            is True
        )
        assert (
            FlexRoleManager.check_compatibility(
                self._p2p["actor2"], self._p2p["actor5"]
            )
            is True
        )
        assert (
            FlexRoleManager.check_compatibility(
                self._p2p["actor3"], self._p2p["actor4"]
            )
            is True
        )
        assert (
            FlexRoleManager.check_compatibility(
                self._p2p["actor3"], self._p2p["actor5"]
            )
            is True
        )
        assert (
            FlexRoleManager.check_compatibility(
                self._p2p["actor4"], self._p2p["actor5"]
            )
            is True
        )
        assert (
            FlexRoleManager.check_compatibility(
                self._p2p["actor5"], self._p2p["actor3"]
            )
            is True
        )
