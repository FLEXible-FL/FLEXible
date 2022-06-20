from flex.data.flex_dataset import FlexDataObject


class FlexDataDistribution(object):
    __create_key = object()

    def __init__(self, create_key: object = None) -> None:
        assert (
            create_key == FlexDataDistribution.__create_key
        ), """FlexDataDistribution objects must be created using FlexDataDistribution.from_state or
        FlexDataDistribution.iid_distribution"""

    @classmethod
    def from_state(cls, data: FlexDataObject = None, state=None):
        """This function prepare the data from a centralized data structure to a federated one.
        It will run diffetent functions to federate the data.

        Args:
            data (FlexDataObject): Centralized dataset represented as a FlexDataObject.
            state (FlexState): FlexState with the configuration to federate the dataset.

        Returns:
            federated_dataset (FederatedFlexDataObject): The dataset federated.
        """
        # TODO: Once FlexState is finished, continue with the development of the class.
        # return FederatedFlexDataObject()

    @classmethod
    def iid_distribution(cls, data: FlexDataObject = None, n_clients: int = 2):
        """Function to create a FlexDataset for an IID experiment.

        Args:
            data (FlexDataObject): Centralizaed dataset represented as a FlexDataObject.
            n_clients (int): Number of clients in the Federated Learning experiment. Default 2.

        Returns:
            federated_dataset (FederatedFlexDatasetObject): Federated Dataset,
        """
        # TODO: Once FlexState is finished, and other functions to create the FlexDataset
        # are finished too, continue with this class.
        # return FederatedFlexDataObject()
