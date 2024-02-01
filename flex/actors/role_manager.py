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


from flex.actors.role import FlexRole


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
        -----
            role (Role): role to be checked

        Returns:
        --------
            bool: wheter the not role is a client role
        """
        return role in cls.client_wannabe

    @classmethod
    def can_comm_with_client(cls, role: FlexRole) -> bool:
        """Method to ensure that role can establish a communication with
        a client role.

        Args:
        -----
            role (Role): role to be checked

        Returns:
        --------
            bool: whether or not role can communicate with a client role
        """
        return role in cls.client_allowed_comm

    @classmethod
    def is_aggregator(cls, role: FlexRole) -> bool:
        """Method to check whether a role is an aggregator role.

        Args:
        -----
            role (Role): role to be checked

        Returns:
        --------
            bool: wheter the not role is an aggregator role
        """
        return role in cls.aggregator_wannabe

    @classmethod
    def can_comm_with_aggregator(cls, role: FlexRole) -> bool:
        """Method to ensure that role can establish a communication with
        an aggregator role.

        Args:
        -----
            role (Role): role to be checked

        Returns:
        --------
            bool: whether or not role can communicate with a aggregator role
        """
        return role in cls.aggregator_allowed_comm

    @classmethod
    def is_server(cls, role: FlexRole) -> bool:
        """Method to check whether a role is a server role.

        Args:
        -----
            role (Role): role to be checked

        Returns:
        --------
            bool: wheter the not role is a server role
        """
        return role in cls.server_wannabe

    @classmethod
    def can_comm_with_server(cls, role: FlexRole) -> bool:
        """Method to ensure that role can establish a communication with
        a server role.

        Args:
        -----
            role (Role): role to be checked

        Returns:
        --------
            bool: whether or not role can communicate with a server role
        """
        return role in cls.server_allowed_comm

    @classmethod
    def check_compatibility(cls, role1: FlexRole, role2: FlexRole) -> bool:
        """Method used to ensure that it is possible to communicate from role1
        to role2, note that the communication from role2 to role1 is not checked.

        Args:
        -----
            role1 (Role): role which establishes communication with role2
            role2 (Role): role which receives communication from role1

        Returns:
        --------
            bool: whether or not the communication from role1 to role2 is allowed.
        """
        return any(
            [
                cls.is_client(role1) and cls.can_comm_with_client(role2),
                cls.is_aggregator(role1) and cls.can_comm_with_aggregator(role2),
                cls.is_server(role1) and cls.can_comm_with_server(role2),
            ]
        )
