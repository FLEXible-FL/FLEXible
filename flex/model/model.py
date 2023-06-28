import collections


class FlexModel(collections.UserDict):
    """Class that represents a model owned by each node in a Federated Experiment.
    The model must be storaged using the key 'model' because of the decorators,
    as we assume that the model will be using that key.
    """

    pass
