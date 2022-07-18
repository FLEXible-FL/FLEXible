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
    client_mask = (
        Role.aggregator,
        Role.aggregator_client,
        Role.server_aggregator,
        Role.server_aggregator_client,
    )
    aggregator_mask = (
        Role.aggregator,
        Role.aggregator_client,
        Role.server,
        Role.server_aggregator,
        Role.server_aggregator_client,
    )
    server_mask = (
        Role.client,
        Role.server_client,
        Role.aggregator_client,
        Role.server_aggregator_client,
    )

    client_wannabe = (
        Role.client,
        Role.server_client,
        Role.aggregator_client,
        Role.server_aggregator_client,
    )
    aggregator_wannabe = (
        Role.client,
        Role.server_client,
        Role.aggregator_client,
        Role.server_aggregator_client,
    )
    server_wannabe = (
        Role.server,
        Role.server_client,
        Role.server_aggregator,
        Role.server_aggregator_client,
    )

    @classmethod
    def isClient(cls, role: Role) -> bool:
        return role in cls.client_wannabe

    @classmethod
    def isAggregator(cls, role: Role) -> bool:
        return role in cls.aggregator_wannabe

    @classmethod
    def isServer(cls, role: Role) -> bool:
        return role in cls.server_wannabe

    @classmethod
    def check_compatibility(cls, role1: Role, role2: Role) -> bool:
        return any(
            [
                cls.isClient(role1) and (role2 in cls.client_mask),
                cls.isAggregator(role1) and (role2 in cls.aggregator_mask),
                cls.isServer(role1) and (role2 in cls.server_mask),
            ]
        )


class Actors(UserDict):
    """ """

    def check_compatibility(self, key1: Hashable, key2: Hashable):
        return RoleManager.check_compatibility(self[key1], self[key2])

    def __setitem__(self, key: Hashable, item: Role) -> None:
        self.data[key] = item
