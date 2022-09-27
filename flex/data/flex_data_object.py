from dataclasses import dataclass, field
from typing import Optional

import numpy.typing as npt


@dataclass(frozen=True)
class FlexDataObject:
    """Class used to represent the dataset from a client in a Federated Learning enviroment.

    Attributes
    ----------
    _X_data: numpy.typing.ArrayLike
        A numpy.array containing the data for the client.
    _y_data: numpy.typing.ArrayLike
        A numpy.array containing the labels for the training data. Can be None if working
        on an unsupervised learning task. Default None.
    """

    X_data: npt.NDArray = field(init=True)
    y_data: Optional[npt.NDArray] = field(default=None, init=True)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, pos):
        if self.y_data is None:
            return FlexDataObject(self.X_data[pos], None)
        else:
            return FlexDataObject(
                self.X_data[pos],
                self.y_data[pos[0]] if isinstance(pos, tuple) else self.y_data[pos],
            )

    def __iter__(self):
        return zip(
            self.X_data,
            self.y_data if self.y_data is not None else [None] * len(self.X_data),
        )

    def validate(self):
        """Function that checks whether the object is correct or not."""
        if self.y_data is not None and len(self.X_data) != len(self.y_data):
            raise ValueError(
                f"X_data and y_data must have equal lenght. X_data has {len(self.X_data)} elements and y_data has {len(self.y_data)} elements."
            )
        if self.y_data is not None and self.y_data.ndim > 1:
            raise ValueError(
                "y_data is multidimensional and we only support unidimensional labels."
            )
