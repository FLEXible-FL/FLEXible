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
import unittest

import pytest

from flex.model import FlexModel


class TestFlexModel(unittest.TestCase):
    def test_getter_actor_id(self):
        flex_model = FlexModel()
        flex_model.actor_id = "server"
        assert flex_model.actor_id == "server"

    def test_setter_actor_id(self):
        flex_model = FlexModel()
        flex_model.actor_id = "server"
        with pytest.raises(PermissionError):
            flex_model.actor_id = "client_1"

    def test_deleter_actor_id(self):
        flex_model = FlexModel()
        flex_model.actor_id = "server"
        del flex_model.actor_id
        assert flex_model.actor_id == "server"
