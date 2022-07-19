from collections import UserDict
from enum import Enum
from typing import Hashable


class Role(Enum):
    client = 1
    aggregator = 2
    server = 3
    aggregator_client = 4
    server_client = 5
    server_aggregator = 6
    server_aggregator_client = 7


class RoleManager:
    client_allowed_comm = {
        Role.aggregator,
        Role.aggregator_client,
        Role.server_aggregator,
        Role.server_aggregator_client,
    }
    aggregator_allowed_comm = {
        Role.aggregator,
        Role.aggregator_client,
        Role.server,
        Role.server_aggregator,
        Role.server_aggregator_client,
    }
    server_allowed_comm = {
        Role.client,
        Role.server_client,
        Role.aggregator_client,
        Role.server_aggregator_client,
    }

    client_wannabe = {
        Role.client,
        Role.server_client,
        Role.aggregator_client,
        Role.server_aggregator_client,
    }
    aggregator_wannabe = {
        Role.client,
        Role.server_client,
        Role.aggregator_client,
        Role.server_aggregator_client,
    }
    server_wannabe = {
        Role.server,
        Role.server_client,
        Role.server_aggregator,
        Role.server_aggregator_client,
    }

    @classmethod
    def is_client(cls, role: Role) -> bool:
        return role in cls.client_wannabe

    @classmethod
    def can_comm_with_client(cls, role: Role) -> bool:
        return role in cls.client_allowed_comm

    @classmethod
    def is_aggregator(cls, role: Role) -> bool:
        return role in cls.aggregator_wannabe

    @classmethod
    def can_comm_with_aggregator(cls, role: Role) -> bool:
        return role in cls.aggregator_allowed_comm

    @classmethod
    def is_server(cls, role: Role) -> bool:
        return role in cls.server_wannabe

    @classmethod
    def can_comm_with_server(cls, role: Role) -> bool:
        return role in cls.server_allowed_comm

    @classmethod
    def check_compatibility(cls, role1: Role, role2: Role) -> bool:
        return any(
            [
                cls.is_client(role1) and cls.can_comm_with_client(role2),
                cls.is_aggregator(role1) and cls.can_comm_with_aggregator(role2),
                cls.is_server(role1) and cls.can_comm_with_server(role2),
            ]
        )


class Actors(UserDict):
    """ """

    def check_compatibility(self, key1: Hashable, key2: Hashable):
        return RoleManager.check_compatibility(self[key1], self[key2])

    def __setitem__(self, key: Hashable, item: Role) -> None:
        self.data[key] = item
