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
