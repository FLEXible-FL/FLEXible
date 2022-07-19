from typing import Callable

from flex.data.flex_dataset import FlexDataset
from flex.pool.actors import Actors, Role


class FlexPoolManager(object):
    """ """

    def __init__(
        self,
        flex_data: FlexDataset = None,
        flex_actors: Actors = None,
        flex_model=None,
        dropout_rate: float = None,
    ) -> None:
        self._actors = flex_actors  # Actors
        self._data = flex_data  # FlexDataset
        self._models = flex_model  # Add when models are finished
        self._dr_rate = dropout_rate  # Connection dropout rate

    def filter(self, func: Callable):
        pass

    def validate(self):
        """
        Validar:
        - Los actores con rol de cliente deben aparecer (mismo id) en self._data.
        - Debe haber, al menos, un actor servidor-aggregador. Puede estar representado por dos actores.
        """
        pass

    @classmethod
    def client_server_architecture(
        cls, fed_dataset: FlexDataset, dropout_rate: float = None
    ):
        actors = Actors({actor_id: Role.client for actor_id in fed_dataset.keys()})
        actors["server"] = Role.server_aggregator
        return cls(
            flex_data=fed_dataset,
            flex_actors=actors,
            flex_model=None,
            dropout_rate=dropout_rate,
        )

    @classmethod
    def p2p_architecture(cls, fed_dataset: FlexDataset, dropout_rate: float = None):
        return cls(
            flex_data=fed_dataset,
            flex_actors=cls.__create_actors_all_privileges(fed_dataset.keys()),
            flex_model=None,
            dropout_rate=dropout_rate,
        )

    @classmethod
    def __create_actors_all_privileges(cls, actors_ids):
        return Actors(
            {actor_id: Role.server_aggregator_client for actor_id in actors_ids}
        )
