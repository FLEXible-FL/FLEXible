import warnings
from dataclasses import dataclass
from typing import Hashable, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from flex.data import FlexDataObject


@dataclass
class FlexDatasetConfig:
    """Class used to represent a configuration to federate a centralized dataset.

    Attributes
    ----------
    seed: Optional[int]
        Seed used to make the federated dataset generated with this configuration reproducible. Default None
    n_clients: int
        Number of clients to split a centralized dataset. Default 2
    weights: Optional[npt.NDArray]
        A numpy.array which provides the proportion of data to give to each client. Default None.
    replacement: bool
        Whether the samping procedure used to split a centralized dataset is with replacement or not. Default True
    classes_per_client: Optional[Union[int, npt.NDArray, Tuple[int]]]
        Classes to assign to each client, if provided as an int, it is the number classes per client, if provided as a \
        tuple of ints, it establishes a mininum and a maximum of number of classes per client, a random number sampled \
        in such interval decides the number of classes of each client. If provided as a list, it establishes the classes \
        assigned to each client.
    features_per_client: Optional[Union[int, npt.NDArray, Tuple[int]]]
        Features to assign to each client, it share the same interface as classes_per_client.
    client_names: Optional[List[Hashable]]
        Names to identifty each client, if not provided clients will be indexed using integers. Default None.
    """

    seed: Optional[int] = None
    n_clients: int = 2
    weights: Optional[npt.NDArray] = None
    replacement: bool = True
    classes_per_client: Optional[Union[int, npt.NDArray, Tuple[int]]] = None
    features_per_client: Optional[Union[int, npt.NDArray, Tuple[int]]] = None
    client_names: Optional[List[Hashable]] = None

    def validate(self, ds: FlexDataObject):
        """ This function checks whether the configuration to federate a dataset is correct and it is compatible with\
            a given centralized dataset.

        Args:
            ds (FlexDataObject): Centralized dataset represented as a FlexDataObject.

        """
        self.__validate_clients_and_weights()
        if self.classes_per_client is not None and self.features_per_client is not None:
            raise ValueError(
                "classes_per_client and features_per_client are mutually exclusive, provide only one."
            )
        elif self.classes_per_client is not None:
            self.__validate_classes_per_client(ds)
        elif self.features_per_client is not None:
            self.__validate_features_per_class(ds)

    def __validate_clients_and_weights(self):
        if self.n_clients < 2:
            raise ValueError(
                f"The number of clients must be greater or equal to 2, but n_clients={self.n_clients}"
            )
        if self.client_names is not None and len(self.client_names) != self.n_clients:
            raise ValueError(
                f"The number of clients must equal the number of client names, but n_clients={self.n_clients} and len(client_names)={len(self.client_names)}"
            )
        if self.weights is not None and self.n_clients != len(self.weights):
            raise ValueError(
                f"The number of weights must equal the number of clients, but n_clients={self.n_clients} and len(weights)={len(self.weights)}."
            )
        if self.weights is not None and max(self.weights) > 1:
            raise ValueError(
                "Provided weights contains an element greater than 1, we do not allow sampling more than one time the entire dataset per client."
            )

    def __validate_classes_per_client(self, ds: FlexDataObject):
        if isinstance(self.classes_per_client, int):
            if self.classes_per_client <= 0 or self.classes_per_client > len(
                np.unique(ds.y_data)
            ):
                raise ValueError(
                    f"classes_per_client if provided as an integer, it must be greater than 0 and smaller than the number of classes in the FlexDataObject, which is {len(np.unique(ds.y_data))}."
                )
        elif isinstance(self.classes_per_client, tuple):
            if len(self.classes_per_client) != 2:
                raise ValueError(
                    f"classes_per_client if provided as a tuple, it must have two elements, mininum number of classes per client and maximum number of classes per client, but classes_per_client={self.classes_per_client}."
                )
            elif self.classes_per_client[0] > self.classes_per_client[1]:
                raise ValueError(
                    f"classes_per_client if provided as a tuple, it must have two elements, mininum number of features per client and maximum number of features per client, but features_per_client={self.features_per_client}."
                )
            elif not self.replacement and self.classes_per_client[
                1
            ] * self.n_clients > len(np.unique(ds.y_data)):
                warnings.warn(
                    "The minimum number of classes_per_client migth not be enforced if each client has the maximum number of classes.",
                    RuntimeWarning,
                )

        elif self.n_clients != len(self.classes_per_client):
            raise ValueError(
                "classes_per_client if provided as a list o np.ndarray, its length and n_clients must equal."
            )

    def __validate_features_per_class(self, ds: FlexDataObject):
        if not self.replacement:
            raise ValueError(
                "By setting replacement to False and specifying features_per_client, clients will not share any data instances."
            )
        if isinstance(self.features_per_client, int):
            if (
                self.features_per_client <= 0
                or self.features_per_client > ds.X_data.shape[1]
            ):
                raise ValueError(
                    f"features_per_client if provided as an integer, it must be greater than 0 and smaller than the number of classes in the FlexDataObject, which is {len(np.unique(ds.y_data))}."
                )
        elif isinstance(self.features_per_client, tuple):
            if len(self.features_per_client) != 2:
                raise ValueError(
                    f"features_per_client if provided as a tuple, it must have two elements, mininum number of features per client and maximum number of features per client, but features_per_client={self.features_per_client}."
                )
            elif self.features_per_client[0] > self.features_per_client[1]:
                raise ValueError(
                    f"features_per_client if provided as a tuple, it must have two elements, mininum number of features per client and maximum number of features per client, but features_per_client={self.features_per_client}."
                )
        elif self.n_clients != len(self.features_per_client):
            raise ValueError(
                "features_per_client if provided as a list o np.ndarray, its length and n_clients must equal."
            )
