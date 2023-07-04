import collections
from typing import Hashable


class FlexModel(collections.UserDict):
    """Class that represents a model owned by each node in a Federated Experiment.
    The model must be storaged using the key 'model' because of the decorators,
    as we assume that the model will be using that key.
    """

    __actor_id: Hashable = None

    @property
    def actor_id(self):
        return self.__actor_id

    @actor_id.setter
    def actor_id(self, value):
        if self.__actor_id is None:
            self.__actor_id = value
        else:
            raise PermissionError("The property actor_id cannot be updated.")

    @actor_id.deleter
    def actor_id(self):
        ...
