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


import collections
from typing import Hashable


class FlexModel(collections.UserDict):
    """Class that represents a model owned by each node in a Federated Experiment.
    The model must be storaged using the key 'model' because of the decorators,
    as we assume that the model will be using that key.
    """

    __actor_id: Hashable = None

    @property
    def actor_id(self):
        return self.__actor_id

    @actor_id.setter
    def actor_id(self, value):
        if self.__actor_id is None:
            self.__actor_id = value
        else:
            raise PermissionError("The property actor_id cannot be updated.")

    @actor_id.deleter
    def actor_id(self):
        ...
