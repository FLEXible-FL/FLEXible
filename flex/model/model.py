from typing import Any, Hashable
from copy import deepcopy


class FlexModel(dict):
    """Class that represents a model owned by each node in a Federated Experiment.
    It is important to note that the model key can only be used once.
    """

    one_use_keys = ["model"]

    def __setitem__(self, key: Hashable, value: Any) -> None:
        if key in self.one_use_keys and key in self:
            raise KeyError(f"{key} key can not be overwritten.")
        else:
            dict.__setitem__(self, key, value)

    def __getattr__(self, key: Hashable):
        try:
            return self[key]
        except KeyError as k:
            raise KeyError(k) from k

    def __setattr__(self, key: Hashable, value: Any):
        if key in self.one_use_keys and key in self:
            raise KeyError(f"{key} key can not be overwritten.")
        else:
            self[key] = value

    def __delattr__(self, key: Hashable):
        try:
            del self[key]
        except KeyError as k:
            raise KeyError(k) from k

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.items():
            setattr(result, k, deepcopy(v, memo))
        return result
