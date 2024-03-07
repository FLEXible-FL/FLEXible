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
from typing import Iterable

from flex.actors.actors import FlexActors, FlexRole


def client_server_architecture(clients_ids: Iterable, server_id: str = "server"):
    """Method to create a client-server architeture from an Iterable of clients
    ids given, and a server id (optional).

    This method will assing to each id from the Iterable the client-role,
    and will create a new actor that will be the server-aggregator that will
    orchestrate the learning phase.

    Args:
    -----
        clients_ids (Iterable): List with the IDs for the clients
        server_id (str, optional): ID for the server actor. Defaults to None.

    Returns:
    --------
        FlexActors: The actors with their roles assigned.
    """
    actors = FlexActors()

    for client_id in clients_ids:
        actors[client_id] = FlexRole.client

    actors[server_id] = FlexRole.server_aggregator

    return actors


def p2p_architecture(nodes_ids: list):
    """Method to create a peer-to-peer (p2p) architecture from an Iterable of
    nodes given.

    This method will assing all roles (client-aggregator-server) to every id from
    the Iterable, so each participant in the learning phase can act as client,
    aggregator and server.

    Args:
    -----
        clients_ids (Iterable): Iterable with the clients ids

    Returns:
    --------
        FlexActors: Actors with their role assigned.
    """
    actors = FlexActors()

    for client_id in nodes_ids:
        actors[client_id] = FlexRole.server_aggregator_client

    return actors
