from collections import UserDict
from typing import Any, Hashable


class FlexModel(UserDict):
    """Class that represents a model owned by each node in a Federated Experiment.
    It is important to note that the model key can only be used once.

    Attributes
    ----------
    data (collections.UserDict): The structure is a dictionary

    """

    one_use_keys = ["model"]

    def __setitem__(self, key: Hashable, item: Any) -> None:
        if key in self.one_use_keys and key in self:
            raise KeyError(f"{key} key can not be overwritten.")
        else:
            self.data[key] = item
