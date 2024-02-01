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
