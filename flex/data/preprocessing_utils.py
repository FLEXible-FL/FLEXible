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


def normalize(node_dataset, *args, **kwargs):
    """Function that normalizes federated data.

    Args:
        node_dataset (Dataset): node_dataset  to normalize the data.

    Returns:
        Dataset: Returns the node_dataset with the X_data property normalized.
    """
    X_data = node_dataset.X_data.to_numpy()
    norms = np.linalg.norm(X_data, axis=0)
    norms = np.where(norms == 0, np.finfo(X_data.dtype).eps, norms)
    new_X_data = X_data / norms
    return Dataset.from_array(new_X_data, node_dataset.y_data.to_numpy())


def one_hot_encoding(node_dataset, *args, **kwargs):
    """Function that apply one hot encoding to the labels of a node_dataset.

    Args:
        node_dataset (Dataset): node_dataset to which apply one hot encode to her labels.

    Raises:
        ValueError: Raises value error if n_labels is not given in the kwargs argument.

    Returns:
        Dataset: Returns the node_dataset with the y_data property updated.
    """
    if "n_labels" not in kwargs:
        raise ValueError(
            "No number of labels given. The parameter n_labels must be given through kwargs."
        )
    y_data = node_dataset.y_data.to_numpy()
    n_labels = int(kwargs["n_labels"])
    one_hot_labels = np.zeros((y_data.size, n_labels))
    one_hot_labels[np.arange(y_data.size), y_data] = 1
    new__y_data = one_hot_labels
    return Dataset(
        X_data=deepcopy(node_dataset.X_data),
        y_data=LazyIndexable(new__y_data, len(new__y_data)),
    )
