<<<<<<< HEAD
# Copyright 2023 Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections import UserDict
=======
from collections import OrderedDict
>>>>>>> 603436d (replace UserDict with OrderedDict)
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
        with the clients ids as keys and Roles as a value.
    """

    def check_compatibility(self, key1: Hashable, key2: Hashable) -> bool:
        """Method to ensure that it is possible to establish communication
        between two actors, according to their roles. Note that the communication
        is stablished from node with key1 to node with key2. Communication from node
        with key2 to node with key1 is not checked.

        Args:
            key1 (Hashable): id used to identify a node. This node is supposed to start communication
            from itself to node with key2.
            key2 (Hashable): id used to identify a node. This node is suppored to receive communication
            from node with key1.

        Returns:
            bool: whether or not the communication is allowed.
        """
        return FlexRoleManager.check_compatibility(self[key1], self[key2])

    def __setitem__(self, key: Hashable, item: FlexRole) -> None:
        super().__setitem__(key, item)
