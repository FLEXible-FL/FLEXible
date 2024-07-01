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
from typing import List

import numpy as np
from proto.tensor_pb2 import Tensor


def toTensorList(tensors: List[np.ndarray]) -> List[Tensor]:
    rv = []
    for tensor in tensors:
        new_t = np.array(tensor, dtype=np.float32)
        rv.append(Tensor(shape=list(new_t.shape), data=new_t.tobytes()))
    return rv


def toNumpyArray(tensors: List[Tensor]) -> List[np.ndarray]:
    rv = []
    for tensor in tensors:
        shape = tuple(tensor.shape)
        t = np.frombuffer(tensor.data, dtype=np.float32).reshape(shape)
        rv.append(t)
    return rv
