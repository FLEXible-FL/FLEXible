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
from collections import OrderedDict
from typing import Hashable

from flex.actors import FlexRole, FlexRoleManager


class FlexActors(OrderedDict):
    """Class that represents roles assigned to each node in a Federated Experiment.
    Roles are designed to restrict communications between nodes. It is important
    to note that Roles are not mutually exclusive, that is, a node can have multiple
    Roles.

    Attributes
    ----------
    data (collections.OrderedDict): The structure is a dictionary
        with the node ids as keys and Roles as a value.
    """

    def check_compatibility(self, key1: Hashable, key2: Hashable) -> bool:
        """Method to ensure that it is possible to establish communication
        between two actors, according to their roles. Note that the communication
        is stablished from node with key1 to node with key2. Communication from node
        with key2 to node with key1 is not checked.

        Args:
        -----
            key1 (Hashable): id used to identify a node. This node is supposed to start communication
            from itself to node with key2.
            key2 (Hashable): id used to identify a node. This node is suppored to receive communication
            from node with key1.

        Returns:
        --------
            bool: whether or not the communication is allowed.
        """
        return FlexRoleManager.check_compatibility(self[key1], self[key2])

    def __setitem__(self, key: Hashable, item: FlexRole) -> None:
        super().__setitem__(key, item)
