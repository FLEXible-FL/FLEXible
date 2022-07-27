from collections import UserDict
from enum import unique, Enum
from typing import Hashable

@unique
class FlexRole(Enum):
    """Enum which contains all possible roles:
        - Basic roles: client, server or aggregator
        - Composite roles: aggregator_client, server_client, server_aggregator, server_aggregator_client
    Note that composite roles are designed to represented a combination of Basic roles.
    """

    client = 1
    aggregator = 2
    server = 3
    aggregator_client = 4
    server_client = 5
    server_aggregator = 6
    server_aggregator_client = 7


class FlexRoleManager:
    """Class used to check allowed communications between
    different roles.
    """

    client_allowed_comm = {
        FlexRole.aggregator,
        FlexRole.aggregator_client,
        FlexRole.server_aggregator,
        FlexRole.server_aggregator_client,
    }
    aggregator_allowed_comm = {
        FlexRole.aggregator,
        FlexRole.aggregator_client,
        FlexRole.server,
        FlexRole.server_aggregator,
        FlexRole.server_client,
        FlexRole.server_aggregator_client,
    }
    server_allowed_comm = {
        FlexRole.client,
        FlexRole.aggregator_client,
        FlexRole.server_client,
        FlexRole.server_aggregator,
        FlexRole.server_aggregator_client,
    }

    client_wannabe = {
        FlexRole.client,
        FlexRole.server_client,
        FlexRole.aggregator_client,
        FlexRole.server_aggregator_client,
    }
    aggregator_wannabe = {
        FlexRole.aggregator,
        FlexRole.server_aggregator,
        FlexRole.aggregator_client,
        FlexRole.server_aggregator_client,
    }
    server_wannabe = {
        FlexRole.server,
        FlexRole.server_client,
        FlexRole.server_aggregator,
        FlexRole.server_aggregator_client,
    }

    @classmethod
    def is_client(cls, role: FlexRole) -> bool:
        """Method to check whether a role is a client role.

        Args:
            role (Role): role to be checked

        Returns:
            bool: wheter the not role is a client role
        """
        return role in cls.client_wannabe

    @classmethod
    def can_comm_with_client(cls, role: FlexRole) -> bool:
        """Method to ensure that role can establish a communication with
        a client role.

        Args:
            role (Role): role to be checked

        Returns:
            bool: whether or not role can communicate with a client role
        """
        return role in cls.client_allowed_comm

    @classmethod
    def is_aggregator(cls, role: FlexRole) -> bool:
        """Method to check whether a role is an aggregator role.

        Args:
            role (Role): role to be checked

        Returns:
            bool: wheter the not role is an aggregator role
        """
        return role in cls.aggregator_wannabe

    @classmethod
    def can_comm_with_aggregator(cls, role: FlexRole) -> bool:
        """Method to ensure that role can establish a communication with
        an aggregator role.

        Args:
            role (Role): role to be checked

        Returns:
            bool: whether or not role can communicate with a aggregator role
        """
        return role in cls.aggregator_allowed_comm

    @classmethod
    def is_server(cls, role: FlexRole) -> bool:
        """Method to check whether a role is a server role.

        Args:
            role (Role): role to be checked

        Returns:
            bool: wheter the not role is a server role
        """
        return role in cls.server_wannabe

    @classmethod
    def can_comm_with_server(cls, role: FlexRole) -> bool:
        """Method to ensure that role can establish a communication with
        a server role.

        Args:
            role (Role): role to be checked

        Returns:
            bool: whether or not role can communicate with a server role
        """
        return role in cls.server_allowed_comm

    @classmethod
    def check_compatibility(cls, role1: FlexRole, role2: FlexRole) -> bool:
        """Method used to ensure that it is possible to communicate from role1
        to role2, note that the communication from role2 to role1 is not checked.

        Args:
            role1 (Role): role which establishes communication with role2
            role2 (Role): role which receives communication from role1

        Returns:
            bool: whether or not the communication from role1 to role2 is allowed.
        """
        return any(
            [
                cls.is_client(role1) and cls.can_comm_with_client(role2),
                cls.is_aggregator(role1) and cls.can_comm_with_aggregator(role2),
                cls.is_server(role1) and cls.can_comm_with_server(role2),
            ]
        )


class FlexActors(UserDict):
    """Class that represents roles assigned to each node in a Federated Experiment.
    Roles are designed to restrict communications between nodes. It is important
    to note that Roles are not mutually exclusive, that is, a node can have multiple
    Roles.

    Attributes
    ----------
    data (collections.UserDict): The structure is a dictionary
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
        self.data[key] = item
