from dataclasses import dataclass
from typing import Hashable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt


@dataclass
class FlexDatasetConfig:
    """Class used to represent a configuration to federate a centralized dataset.
    The following table shows the compatiblity of each option:

    | Options compatibility   | **n_clients** | **client_names** | **weights** | **weights_per_class** | **replacement** | **classes_per_client** | **features_per_client** | **indexes_per_client** |
    |-------------------------|---------------|------------------|-------------|-----------------------|-----------------|------------------------|-------------------------|------------------------|
    | **n_clients**           | -             | Y                | Y           | Y                     | Y               | Y                      | Y                       | Y                      |
    | **client_names**        | Y             | -                | Y           | Y                     | Y               | Y                      | Y                       | Y                      |
    | **weights**             | Y             | Y                | -           | N                     | Y               | Y                      | Y                       | N                      |
    | **weights_per_class**   | Y             | Y                | N           | -                     | Y               | N                      | N                       | N                      |
    | **replacement**         | Y             | Y                | Y           | Y                     | -               | Y                      | Y                       | N                      |
    | **classes_per_client**  | Y             | Y                | Y           | N                     | Y               | -                      | N                       | N                      |
    | **features_per_client** | Y             | Y                | Y           | N                     | Y               | N                      | -                       | N                      |
    | **indexes_per_client**  | Y             | Y                | N           | N                     | N               | N                      | N                       | -                      |

    Attributes
    ----------
    seed: Optional[int]
        Seed used to make the federated dataset generated reproducible with this configuration. Default None.
    n_clients: Optional[int]
        Number of clients among which to split the centralized dataset. If client_names is also given, we consider the number \
        of clients to be the minimun between n_clients and the length of client_names. Default None.
    client_names: Optional[List[Hashable]]
        Names to identifty each client, if not provided clients will be indexed using integers. If n_clients is also \
        given, we consider the number of clients to be the minimun of n_clients and the length of client_names. Default None.
    weights: Optional[npt.NDArray]
        A numpy.array which provides the proportion of data to give to each client. It is not compatible with weights_per_class. Default None.
    weights_per_class: Optional[npt.NDArray]
        A numpy.array which provides the proportion of data to give to each client and class of the dataset to federate. \
        We expect a bidimensional array of shape (n, m) where "n" is the number of clients and "m" is the number of classes of \
        the dataset to federate. It is not compatible with weights. Default None.
    replacement: bool
        Whether the samping procedure used to split a centralized dataset is with replacement or not. Default True
    classes_per_client: Optional[Union[int, npt.NDArray, Tuple[int]]]
        Classes to assign to each client, if provided as an int, it is the number classes per client, if provided as a \
        tuple of ints, it establishes a mininum and a maximum of number of classes per client, a random number sampled \
        in such interval decides the number of classes of each client. If provided as a list, it establishes the classes \
        assigned to each client. It is not compatible with features_per_client. Default None.
    features_per_client: Optional[Union[int, npt.NDArray, Tuple[int]]]
        Features to assign to each client, it share the same interface as classes_per_client. It is not complatible with classes_per_client and weights_per_class \
        Default None.
    indexes_per_client: Optional[npt.NDArray]
        Data indexes to assign to each client, note that this option is incompatible with classes_per_client, \
        features_per_client options. If replacement, weights or weights_per_class are speficied, they are ignored. Default None.
    """

    seed: Optional[int] = None
    n_clients: Optional[int] = None
    client_names: Optional[List[Hashable]] = None
    weights: Optional[npt.NDArray] = None
    weights_per_class: Optional[npt.NDArray] = None
    replacement: bool = True
    classes_per_client: Optional[Union[int, npt.NDArray, Tuple[int]]] = None
    features_per_client: Optional[Union[int, npt.NDArray, Tuple[int]]] = None
    indexes_per_client: Optional[npt.NDArray] = None

    def validate(self):
        """This function checks whether the configuration to federate a dataset is correct."""
        self.__validate_clients_and_weights()
        if self.indexes_per_client is not None:
            self.__validate_indexes_per_client()
        elif (
            self.classes_per_client is not None and self.features_per_client is not None
        ):
            raise ValueError(
                "classes_per_client and features_per_client are mutually exclusive, provide only one."
            )
        elif self.classes_per_client is not None:
            self.__validate_classes_per_client()
        elif self.features_per_client is not None:
            self.__validate_features_per_class()

    def __validate_indexes_per_client(self):
        if self.classes_per_client is not None or self.features_per_client is not None:
            raise ValueError(
                "Indexes_per_client is not compatible with classes_per_client and features_per_client. \
                    If replacement or weights are speficied, they are ignored."
            )
        if (
            self.n_clients is not None
            and len(self.indexes_per_client) != self.n_clients
        ) or (
            self.client_names is not None
            and len(self.indexes_per_client) != len(self.client_names)
        ):
            raise ValueError(
                "The number of provided clients should equal the length of indexes per client."
            )

    def __validate_clients_and_weights(self):
        if self.n_clients is None and self.client_names is None:
            raise ValueError("Either n_clients or client_names must be given.")
        elif (self.n_clients is not None and self.n_clients < 2) or (
            self.client_names is not None and len(self.client_names) < 2
        ):
            raise ValueError("The number of clients must be greater or equal to 2.")

        if self.weights is not None and (
            (
                self.n_clients is not None
                and self.client_names is None
                and self.n_clients != len(self.weights)
            )
            or (
                self.n_clients is None
                and self.client_names is not None
                and len(self.client_names) != len(self.weights)
            )
            or (
                self.n_clients is not None
                and self.client_names is not None
                and min(self.n_clients, len(self.client_names)) != len(self.weights)
            )
        ):
            raise ValueError("The number of weights must equal the number of clients.")

        if self.weights is not None and self.weights_per_class is not None:
            raise ValueError(
                "weights and weights_per_class are not compatible, please provide only one of them."
            )

        if self.weights_per_class is not None and self.classes_per_client is not None:
            raise ValueError(
                "weights_per_class and classes_per_clients are not compatible, please provide only one of them."
            )
        if (
            self.weights_per_class is not None
            and len(np.asarray(self.weights_per_class).shape) != 2
        ):
            raise ValueError(
                (
                    "weights_per_class must be a two dimensional array where the first dimension is the number of clients and the second is the number of classes of the dataset to federate."
                )
            )

        if self.weights_per_class is not None and (
            (
                self.n_clients is not None
                and self.client_names is None
                and self.n_clients != len(self.weights_per_class)
            )
            or (
                self.n_clients is None
                and self.client_names is not None
                and len(self.client_names) != len(self.weights_per_class)
            )
            or (
                self.n_clients is not None
                and self.client_names is not None
                and min(self.n_clients, len(self.client_names))
                != len(self.weights_per_class)
            )
        ):
            raise ValueError(
                "The length of weights_per_class must equal the number of clients."
            )

        if self.weights is not None and max(self.weights) > 1:
            raise ValueError(
                "Provided weights contains an element greater than 1, we do not allow sampling more than one time the entire dataset per client."
            )
        if self.weights is not None and min(self.weights) < 0:
            raise ValueError(
                "Provided weights contains negative numbers, we do not allow that."
            )

    def __validate_classes_per_client(self):
        if isinstance(self.classes_per_client, tuple):
            if len(self.classes_per_client) != 2:
                raise ValueError(
                    f"classes_per_client if provided as a tuple, it must have two elements, mininum number of classes per client and maximum number of classes per client, but classes_per_client={self.classes_per_client}."
                )
        elif not isinstance(self.classes_per_client, int) and (
            (
                self.n_clients is not None
                and self.client_names is None
                and self.n_clients != len(self.classes_per_client)
            )
            or (
                self.client_names is not None
                and self.n_clients is None
                and len(self.client_names) != len(self.classes_per_client)
            )
            or (
                self.n_clients is not None
                and self.client_names is not None
                and min(self.n_clients, len(self.client_names))
                != len(self.classes_per_client)
            )
        ):
            raise ValueError(
                "classes_per_client if provided as a list o np.ndarray, its length and n_clients must equal."
            )

    def __validate_features_per_class(self):
        if not self.replacement:
            raise ValueError(
                "By setting replacement to False and specifying features_per_client, clients will not share any data instances."
            )
        if isinstance(self.features_per_client, tuple):
            if len(self.features_per_client) != 2:
                raise ValueError(
                    f"features_per_client if provided as a tuple, it must have two elements, mininum number of features per client and maximum number of features per client, but features_per_client={self.features_per_client}."
                )
        elif not isinstance(self.features_per_client, int) and (
            (
                self.n_clients is not None
                and self.client_names is None
                and self.n_clients != len(self.features_per_client)
            )
            or (
                self.client_names is not None
                and self.n_clients is None
                and len(self.client_names) != len(self.features_per_client)
            )
            or (
                self.n_clients is not None
                and self.client_names is not None
                and min(self.n_clients, len(self.client_names))
                != len(self.features_per_client)
            )
        ):
            raise ValueError(
                "features_per_client if provided as a list o np.ndarray, its length and n_clients must equal."
            )
