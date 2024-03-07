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
