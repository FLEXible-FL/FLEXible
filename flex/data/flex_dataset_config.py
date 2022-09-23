from dataclasses import dataclass
from typing import Hashable, List, Optional, Tuple, Union

import numpy.typing as npt


@dataclass
class FlexDatasetConfig:
    """Class used to represent a configuration to federate a centralized dataset.

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
        A numpy.array which provides the proportion of data to give to each client. Default None.
    replacement: bool
        Whether the samping procedure used to split a centralized dataset is with replacement or not. Default True
    classes_per_client: Optional[Union[int, npt.NDArray, Tuple[int]]]
        Classes to assign to each client, if provided as an int, it is the number classes per client, if provided as a \
        tuple of ints, it establishes a mininum and a maximum of number of classes per client, a random number sampled \
        in such interval decides the number of classes of each client. If provided as a list, it establishes the classes \
        assigned to each client. Default None.
    features_per_client: Optional[Union[int, npt.NDArray, Tuple[int]]]
        Features to assign to each client, it share the same interface as classes_per_client.
    indexes_per_client: Optional[npt.NDArray]
        Data indexes to assign to each client, note that this option is incompatible with classes_per_client, \
        features_per_client options. If replacement and weights are speficied, they are ignored.
    """

    seed: Optional[int] = None
    n_clients: Optional[int] = None
    client_names: Optional[List[Hashable]] = None
    weights: Optional[npt.NDArray] = None
    replacement: bool = True
    classes_per_client: Optional[Union[int, npt.NDArray, Tuple[int]]] = None
    features_per_client: Optional[Union[int, npt.NDArray, Tuple[int]]] = None
    indexes_per_client: Optional[npt.NDArray] = None

    def validate(self):
        """This function checks whether the configuration to federate a dataset is correct."""
        if self.indexes_per_client is not None and (
            self.classes_per_client is not None or self.features_per_client is not None
        ):
            raise ValueError(
                "Indexes_per_client option is not compatible with classes_per_client and features_per_client. \
                    If replacement and weights are speficied, they are ignored."
            )
        self.__validate_clients_and_weights()
        if self.classes_per_client is not None and self.features_per_client is not None:
            raise ValueError(
                "classes_per_client and features_per_client are mutually exclusive, provide only one."
            )
        elif self.classes_per_client is not None:
            self.__validate_classes_per_client()
        elif self.features_per_client is not None:
            self.__validate_features_per_class()

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
                self.client_names is not None
                and self.n_clients is None
                and len(self.client_names) != len(self.weights)
            )
            or (
                self.n_clients is not None
                and self.client_names is not None
                and min(self.n_clients, len(self.client_names)) != len(self.weights)
            )
        ):
            raise ValueError("The number of weights must equal the number of clients.")

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
