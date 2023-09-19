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


from copy import deepcopy

import numpy as np

from flex.data.dataset import Dataset
from flex.data.lazy_indexable import LazyIndexable


def normalize(client, *args, **kwargs):
    """Function that normalizes the federated data.

    Args:
        client (Dataset): client whether to normalize the data.

    Returns:
        Dataset: Returns the client with the X_data property normalized.
    """
    X_data = client.X_data.to_numpy()
    norms = np.linalg.norm(X_data, axis=0)
    norms = np.where(norms == 0, np.finfo(X_data.dtype).eps, norms)
    new_X_data = X_data / norms
    return Dataset.from_numpy(new_X_data, client.y_data.to_numpy())


def one_hot_encoding(client, *args, **kwargs):
    """Function that apply one hot encoding to the labels of a client.

    Args:
        client (Dataset): client to which apply one hot encode to her classes.

    Raises:
        ValueError: Raises value error if n_classes is not given in the kwargs argument.

    Returns:
        Dataset: Returns the client with the y_data property updated.
    """
    if "n_classes" not in kwargs:
        raise ValueError(
            "No number of classes given. The parameter n_classes must be given through kwargs."
        )
    y_data = client.y_data.to_numpy()
    n_classes = int(kwargs["n_classes"])
    one_hot_classes = np.zeros((y_data.size, n_classes))
    one_hot_classes[np.arange(y_data.size), y_data] = 1
    new_client_y_data = one_hot_classes
    return Dataset(
        X_data=deepcopy(client.X_data),
        y_data=LazyIndexable(new_client_y_data, len(new_client_y_data)),
    )
