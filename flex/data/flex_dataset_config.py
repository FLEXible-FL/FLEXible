from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt

from flex.data import FlexDataObject


@dataclass
class FlexDatasetConfig:
    n_clients: int = 2
    weights: Optional[npt.NDArray] = None
    replacement: bool = True
    classes_per_client: Optional[Union[int, npt.NDArray, Tuple[int]]] = None
    features_per_client: Optional[Union[int, npt.NDArray, Tuple[int]]] = None

    def check(self, ds: FlexDataObject):
        if self.n_clients < 2:
            raise ValueError(
                f"The number of clients must be greater or equal to 2, but n_clients={self.n_clients}"
            )
        if self.weights is not None and self.n_clients != len(self.weights):
            raise ValueError(
                f"The number of weights must equal the number of clients, but n_clients={self.n_clients} and len(weights)={len(self.weights)}."
            )
        if self.classes_per_client is not None and self.features_per_client is not None:
            raise ValueError(
                "classes_per_client and features_per_client are mutually exclusive, provide only one."
            )
        if self.classes_per_client is not None:
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
            else:
                if self.n_clients != len(self.classes_per_client):
                    raise ValueError(
                        "classes_per_client if provided as a list o np.ndarray, its length and n_clients must equal."
                    )
        if self.features_per_client is not None:
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
            else:
                if self.n_clients != len(self.features_per_client):
                    raise ValueError(
                        "features_per_client if provided as a list o np.ndarray, its length and n_clients must equal."
                    )
