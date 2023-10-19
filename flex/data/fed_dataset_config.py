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


from dataclasses import asdict, dataclass
from typing import Hashable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt


class InvalidConfig(ValueError):
    """Raised when the input config is wrong"""

    pass


@dataclass
class FedDatasetConfig:
    """Class used to represent a configuration to federate a centralized dataset.
    The following table shows the compatiblity of each option:

    | Options compatibility   | **n_clients** | **client_names** | **weights** | **weights_per_class** | **replacement** | **classes_per_client** | **features_per_client** | **indexes_per_client** | **group_by_label_index** | **keep_labels** |
    |-------------------------|---------------|------------------|-------------|-----------------------|-----------------|------------------------|-------------------------|------------------------|----------------------|----------------------|
    | **n_clients**           | -             | Y                | Y           | Y                     | Y               | Y                      | Y                       | N                      | N                    | Y                    |
    | **client_names**        | -             | -                | Y           | Y                     | Y               | Y                      | Y                       | Y                      | N                    | Y                    |
    | **weights**             | -             | -                | -           | N                     | Y               | Y                      | Y                       | N                      | N                    | Y                    |
    | **weights_per_class**   | -             | -                | -           | -                     | Y               | N                      | N                       | N                      | N                    | Y                    |
    | **replacement**         | -             | -                | -           | -                     | -               | Y                      | N                       | N                      | N                    | Y                    |
    | **classes_per_client**  | -             | -                | -           | -                     | -               | -                      | N                       | N                      | N                    | Y                    |
    | **features_per_client** | -             | -                | -           | -                     | -               | -                      | -                       | N                      | N                    | Y                    |
    | **indexes_per_client**  | -             | -                | -           | -                     | -               | -                      | -                       | -                      | N                    | Y                    |
    | **group_by_label_index**| -             | -                | -           | -                     | -               | -                      | -                       | -                      | -                    | N                    |
    | **keep_labels**         | -             | -                | -           | -                     | -               | -                      | -                       | -                      | -                    | -                    |

    Attributes
    ----------
    seed: Optional[int]
        Seed used to make the federated dataset generated reproducible with this configuration. Default None.
    n_clients: int
        Number of clients among which to split the centralized dataset. Default 2.
    client_names: Optional[List[Hashable]]
        Names to identifty each client, if not provided clients will be indexed using integers. If n_clients is also \
        given, we consider up to n_clients elements. Default None.
    weights: Optional[npt.NDArray]
        A numpy.array which provides the proportion of data to give to each client. Default None.
    weights_per_class: Optional[npt.NDArray]
        A numpy.array which provides the proportion of data to give to each client and class of the dataset to federate. \
        We expect a bidimensional array of shape (n, m) where "n" is the number of clients and "m" is the number of classes of \
        the dataset to federate. Default None.
    replacement: bool
        Whether the samping procedure used to split a centralized dataset is with replacement or not. Default False
    classes_per_client: Optional[Union[int, npt.NDArray, Tuple[int]]]
        Classes to assign to each client, if provided as an int, it is the number classes per client, if provided as a \
        tuple of ints, it establishes a mininum and a maximum of number of classes per client, a random number sampled \
        in such interval decides the number of classes of each client. If provided as a list of lists, it establishes the classes \
        assigned to each client. Default None.
    features_per_client: Optional[Union[int, npt.NDArray, Tuple[int]]]
        Features to assign to each client, it share the same interface as classes_per_client. Default None.
    indexes_per_client: Optional[npt.NDArray]
        Data indexes to assign to each client. Default None.
    group_by_label_index: Optional[int]
        Index which indicates which feature unique values will be used to generate federated clients. Default None.
    keep_labels: Optional[list[bool]]
        Whether each node keeps or not the labels or y_data
    """

    seed: Optional[int] = None
    n_clients: int = 2
    client_names: Optional[List[Hashable]] = None
    weights: Optional[npt.NDArray] = None
    weights_per_class: Optional[npt.NDArray] = None
    replacement: bool = False
    classes_per_client: Optional[Union[int, npt.NDArray, Tuple[int]]] = None
    features_per_client: Optional[Union[int, npt.NDArray, Tuple[int]]] = None
    indexes_per_client: Optional[npt.NDArray] = None
    group_by_label_index: Optional[int] = None
    keep_labels: Optional[list[bool]] = None

    def _check_incomp(self, dict, option1, option2):
        """This function checks if two options are compatible, if not it raises and exception"""
        cond1 = dict[option1] is not None if option1 != "replacement" else dict[option1]
        cond2 = dict[option2] is not None if option2 != "replacement" else dict[option2]
        if cond1 and cond2:
            raise InvalidConfig(
                f"Options {option1} and {option2} are incompatible, please provide only one."
            )

    def validate(self):
        """This function checks whether the configuration to federate a dataset is correct."""
        self_dict = asdict(self)
        # By default every option is compatible, therefore we only specify incompatibilities
        self._check_incomp(self_dict, "weights", "group_by_label_index")
        self._check_incomp(self_dict, "weights", "weights_per_class")
        self._check_incomp(self_dict, "weights", "indexes_per_client")
        self._check_incomp(self_dict, "weights", "group_by_label_index")
        self._check_incomp(self_dict, "weights_per_class", "indexes_per_client")
        self._check_incomp(self_dict, "weights_per_class", "classes_per_client")
        self._check_incomp(self_dict, "weights_per_class", "features_per_client")
        self._check_incomp(self_dict, "weights_per_class", "indexes_per_client")
        self._check_incomp(self_dict, "weights_per_class", "group_by_label_index")
        self._check_incomp(self_dict, "replacement", "indexes_per_client")
        self._check_incomp(self_dict, "replacement", "group_by_label_index")
        self._check_incomp(self_dict, "classes_per_client", "features_per_client")
        self._check_incomp(self_dict, "classes_per_client", "indexes_per_client")
        self._check_incomp(self_dict, "classes_per_client", "group_by_label_index")
        self._check_incomp(self_dict, "features_per_client", "indexes_per_client")
        self._check_incomp(self_dict, "features_per_client", "group_by_label_index")

        self.__validate_clients_and_weights()
        if self.indexes_per_client is not None:
            self.__validate_indexes_per_client()
        elif self.classes_per_client is not None:
            self.__validate_classes_per_client()
        elif self.features_per_client is not None:
            self.__validate_features_per_client()
        if self.keep_labels is not None:
            self.__validate_keep_labels()

    def __validate_indexes_per_client(self):
        if len(self.indexes_per_client) != self.n_clients:
            raise InvalidConfig(
                "The number of provided clients should equal the length of indexes per client."
            )

    def __validate_keep_labels(self):
        if len(self.keep_labels) != self.n_clients:
            raise InvalidConfig(
                "keep_labels list should have the same length as n_clients."
            )

    def __validate_clients_and_weights(self):
        if self.n_clients < 2:
            raise InvalidConfig(
                "The number of clients must be greater or equal to 2. Default is 2"
            )

        if self.client_names is not None and self.n_clients > len(self.client_names):
            raise InvalidConfig(
                "The number of named clients, client_names, can not be greater than the number of clients, n_clients"
            )

        if self.weights is not None and self.n_clients != len(self.weights):
            raise InvalidConfig(
                "The number of weights must equal the number of clients."
            )

        if (
            self.weights_per_class is not None
            and len(np.asarray(self.weights_per_class).shape) != 2
        ):
            raise InvalidConfig(
                (
                    "weights_per_class must be a two dimensional array where the first dimension is the number of clients and the second is the number of classes of the dataset to federate."
                )
            )
        if self.weights_per_class is not None and self.n_clients != len(
            self.weights_per_class
        ):
            raise InvalidConfig(
                "The length of weights_per_class must equal the number of clients."
            )

        if self.weights is not None and max(self.weights) > 1:
            raise InvalidConfig(
                "Provided weights contains an element greater than 1, we do not allow sampling more than one time the entire dataset per client."
            )
        if self.weights is not None and min(self.weights) < 0:
            raise InvalidConfig(
                "Provided weights contains negative numbers, we do not allow that."
            )

    def __validate_classes_per_client(self):
        if isinstance(self.classes_per_client, tuple):
            if len(self.classes_per_client) != 2:
                raise InvalidConfig(
                    f"classes_per_client if provided as a tuple, it must have two elements, mininum number of classes per client and maximum number of classes per client, but classes_per_client={self.classes_per_client}."
                )
        elif not isinstance(self.classes_per_client, int) and self.n_clients != len(
            self.classes_per_client
        ):
            raise InvalidConfig(
                "classes_per_client if provided as a list o np.ndarray, its length and n_clients must equal."
            )

    def __validate_features_per_client(self):
        if not self.replacement:
            raise InvalidConfig(
                "By setting replacement to False and specifying features_per_client, clients will not share any data instances."
            )
        if isinstance(self.features_per_client, tuple):
            if len(self.features_per_client) != 2:
                raise InvalidConfig(
                    f"features_per_client if provided as a tuple, it must have two elements, mininum number of features per client and maximum number of features per client, but features_per_client={self.features_per_client}."
                )
        elif not isinstance(self.features_per_client, int) and self.n_clients != len(
            self.features_per_client
        ):
            raise InvalidConfig(
                "features_per_client if provided as a list o np.ndarray, its length and n_clients must equal."
            )
