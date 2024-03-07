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

    +-------------------------+-------------+--------------+-------------+-----------------------+-----------------+---------------------+-----------------------+----------------------+--------------------------+-----------------+-------------+
    | Options compatibility   | **n_nodes** | **node_ids** | **weights** | **weights_per_label** | **replacement** | **labels_per_node** | **features_per_node** | **indexes_per_node** | **group_by_label_index** | **keep_labels** | **shuffle** |
    +=========================+=============+==============+=============+=======================+=================+=====================+=======================+======================+==========================+=================+=============+
    | **n_nodes**             | -           | Y            | Y           | Y                     | Y               | Y                   | Y                     | N                    | N                        | Y               | Y           |
    +-------------------------+-------------+--------------+-------------+-----------------------+-----------------+---------------------+-----------------------+----------------------+--------------------------+-----------------+-------------+
    | **node_ids**            | -           | -            | Y           | Y                     | Y               | Y                   | Y                     | Y                    | N                        | Y               | Y           |
    +-------------------------+-------------+--------------+-------------+-----------------------+-----------------+---------------------+-----------------------+----------------------+--------------------------+-----------------+-------------+
    | **weights**             | -           | -            | -           | N                     | Y               | Y                   | Y                     | N                    | N                        | Y               | Y           |
    +-------------------------+-------------+--------------+-------------+-----------------------+-----------------+---------------------+-----------------------+----------------------+--------------------------+-----------------+-------------+
    | **weights_per_label**   | -           | -            | -           | -                     | Y               | N                   | N                     | N                    | N                        | Y               | Y           |
    +-------------------------+-------------+--------------+-------------+-----------------------+-----------------+---------------------+-----------------------+----------------------+--------------------------+-----------------+-------------+
    | **replacement**         | -           | -            | -           | -                     | -               | Y                   | N                     | N                    | N                        | Y               | Y           |
    +-------------------------+-------------+--------------+-------------+-----------------------+-----------------+---------------------+-----------------------+----------------------+--------------------------+-----------------+-------------+
    | **labels_per_node**     | -           | -            | -           | -                     | -               | -                   | N                     | N                    | N                        | Y               | Y           |
    +-------------------------+-------------+--------------+-------------+-----------------------+-----------------+---------------------+-----------------------+----------------------+--------------------------+-----------------+-------------+
    | **features_per_node**   | -           | -            | -           | -                     | -               | -                   | -                     | N                    | N                        | Y               | Y           |
    +-------------------------+-------------+--------------+-------------+-----------------------+-----------------+---------------------+-----------------------+----------------------+--------------------------+-----------------+-------------+
    | **indexes_per_node**    | -           | -            | -           | -                     | -               | -                   | -                     | -                    | N                        | Y               | Y           |
    +-------------------------+-------------+--------------+-------------+-----------------------+-----------------+---------------------+-----------------------+----------------------+--------------------------+-----------------+-------------+
    | **group_by_label_index**| -           | -            | -           | -                     | -               | -                   | -                     | -                    | -                        | N               | Y           |
    +-------------------------+-------------+--------------+-------------+-----------------------+-----------------+---------------------+-----------------------+----------------------+--------------------------+-----------------+-------------+
    | **keep_labels**         | -           | -            | -           | -                     | -               | -                   | -                     | -                    | -                        | -               | Y           |
    +-------------------------+-------------+--------------+-------------+-----------------------+-----------------+---------------------+-----------------------+----------------------+--------------------------+-----------------+-------------+
    | **shuffle**             | -           | -            | -           | -                     | -               | -                   | -                     | -                    | -                        | -               | -           |
    +-------------------------+-------------+--------------+-------------+-----------------------+-----------------+---------------------+-----------------------+----------------------+--------------------------+-----------------+-------------+

    Attributes
    ----------
    seed: Optional[int]
        Seed used to make the federated dataset generated reproducible with this configuration. Default None.
    n_nodes: int
        Number of nodes among which to split a centralized dataset. Default 2.
    shuffle: bool
        If True data is shuffled before being sampled. Default False.
    node_ids: Optional[List[Hashable]]
        Ids to identifty each node, if not provided, nodes will be indexed using integers. If n_nodes is also \
        given, we consider up to n_nodes elements. Default None.
    weights: Optional[npt.NDArray]
        A numpy.array which provides the proportion of data to give to each node. Default None.
    weights_per_label: Optional[npt.NDArray]
        A numpy.array which provides the proportion of data to give to each node and class of the dataset to federate. \
        We expect a bidimensional array of shape (n, m) where "n" is the number of nodes and "m" is the number of labels of \
        the dataset to federate. Default None.
    replacement: bool
        Whether the samping procedure used to split a centralized dataset is with replacement or not. Default False
    labels_per_node: Optional[Union[int, npt.NDArray, Tuple[int]]]
        labels to assign to each node, if provided as an int, it is the number labels per node, if provided as a \
        tuple of ints, it establishes a mininum and a maximum of number of labels per node, a random number sampled \
        in such interval decides the number of labels of each node. If provided as a list of lists, it establishes the labels \
        assigned to each node. Default None.
    features_per_node: Optional[Union[int, npt.NDArray, Tuple[int]]]
        Features to assign to each node, it share the same interface as labels_per_node. Default None.
    indexes_per_node: Optional[npt.NDArray]
        Data indexes to assign to each node. Default None.
    group_by_label_index: Optional[int]
        Index which indicates which feature unique values will be used to generate federated nodes. Default None.
    keep_labels: Optional[list[bool]]
        Whether each node keeps or not the labels or y_data
    """

    seed: Optional[int] = None
    n_nodes: int = 2
    shuffle: bool = False
    node_ids: Optional[List[Hashable]] = None
    weights: Optional[npt.NDArray] = None
    weights_per_label: Optional[npt.NDArray] = None
    replacement: bool = False
    labels_per_node: Optional[Union[int, npt.NDArray, Tuple[int]]] = None
    features_per_node: Optional[Union[int, npt.NDArray, Tuple[int]]] = None
    indexes_per_node: Optional[npt.NDArray] = None
    group_by_label_index: Optional[int] = None
    keep_labels: Optional[List[bool]] = None

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
        self._check_incomp(self_dict, "weights", "weights_per_label")
        self._check_incomp(self_dict, "weights", "indexes_per_node")
        self._check_incomp(self_dict, "weights", "group_by_label_index")
        self._check_incomp(self_dict, "weights_per_label", "indexes_per_node")
        self._check_incomp(self_dict, "weights_per_label", "labels_per_node")
        self._check_incomp(self_dict, "weights_per_label", "features_per_node")
        self._check_incomp(self_dict, "weights_per_label", "indexes_per_node")
        self._check_incomp(self_dict, "weights_per_label", "group_by_label_index")
        self._check_incomp(self_dict, "replacement", "indexes_per_node")
        self._check_incomp(self_dict, "replacement", "group_by_label_index")
        self._check_incomp(self_dict, "labels_per_node", "features_per_node")
        self._check_incomp(self_dict, "labels_per_node", "indexes_per_node")
        self._check_incomp(self_dict, "labels_per_node", "group_by_label_index")
        self._check_incomp(self_dict, "features_per_node", "indexes_per_node")
        self._check_incomp(self_dict, "features_per_node", "group_by_label_index")

        self.__validate_nodes_and_weights()
        if self.indexes_per_node is not None:
            self.__validate_indexes_per_node()
        elif self.labels_per_node is not None:
            self.__validate_labels_per_node()
        elif self.features_per_node is not None:
            self.__validate_features_per_node()
        if self.keep_labels is not None:
            self.__validate_keep_labels()

    def __validate_indexes_per_node(self):
        if len(self.indexes_per_node) != self.n_nodes:
            raise InvalidConfig(
                "The number of provided nodes should equal the length of indexes per node."
            )

    def __validate_keep_labels(self):
        if len(self.keep_labels) != self.n_nodes:
            raise InvalidConfig(
                "keep_labels list should have the same length as n_nodes."
            )

    def __validate_nodes_and_weights(self):
        if self.n_nodes < 2:
            raise InvalidConfig(
                "The number of nodes must be greater or equal to 2. Default is 2"
            )

        if self.node_ids is not None and self.n_nodes > len(self.node_ids):
            raise InvalidConfig(
                "The number of named nodes, node_ids, can not be greater than the number of nodes, n_nodes"
            )

        if self.weights is not None and self.n_nodes != len(self.weights):
            raise InvalidConfig("The number of weights must equal the number of nodes.")

        if (
            self.weights_per_label is not None
            and len(np.asarray(self.weights_per_label).shape) != 2
        ):
            raise InvalidConfig(
                (
                    "weights_per_label must be a two dimensional array where the first dimension is the number of nodes and the second is the number of labels of the dataset to federate."
                )
            )
        if self.weights_per_label is not None and self.n_nodes != len(
            self.weights_per_label
        ):
            raise InvalidConfig(
                "The length of weights_per_label must equal the number of nodes."
            )

        if self.weights is not None and max(self.weights) > 1:
            raise InvalidConfig(
                "Provided weights contains an element greater than 1, we do not allow sampling more than one time the entire dataset per node."
            )
        if self.weights is not None and min(self.weights) < 0:
            raise InvalidConfig(
                "Provided weights contains negative numbers, we do not allow that."
            )

    def __validate_labels_per_node(self):
        if isinstance(self.labels_per_node, tuple):
            if len(self.labels_per_node) != 2:
                raise InvalidConfig(
                    f"labels_per_node if provided as a tuple, it must have two elements, mininum number of labels per node and maximum number of labels per node, but labels_per_node={self.labels_per_node}."
                )
        elif not isinstance(self.labels_per_node, int) and self.n_nodes != len(
            self.labels_per_node
        ):
            raise InvalidConfig(
                "labels_per_node if provided as a list o np.ndarray, its length and n_nodes must equal."
            )

    def __validate_features_per_node(self):
        if not self.replacement:
            raise InvalidConfig(
                "By setting replacement to False and specifying features_per_node, nodes will not share any data instances."
            )
        if isinstance(self.features_per_node, tuple):
            if len(self.features_per_node) != 2:
                raise InvalidConfig(
                    f"features_per_node if provided as a tuple, it must have two elements, mininum number of features per node and maximum number of features per node, but features_per_node={self.features_per_node}."
                )
        elif not isinstance(self.features_per_node, int) and self.n_nodes != len(
            self.features_per_node
        ):
            raise InvalidConfig(
                "features_per_node if provided as a list o np.ndarray, its length and n_nodes must equal."
            )
