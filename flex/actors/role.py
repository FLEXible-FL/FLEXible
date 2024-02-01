from enum import Enum, unique


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
