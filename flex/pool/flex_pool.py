from typing import Callable

from flex.data import FlexDataset
from flex.pool.actors import FlexActors, FlexRole, FlexRoleManager


class FlexPoolManager:
    """ """

    def __init__(
        self,
        flex_data: FlexDataset = None,
        flex_actors: FlexActors = None,
        flex_model=None,
        dropout_rate: float = None,
    ) -> None:
        self._actors = flex_actors  # Actors
        self._data = flex_data  # FlexDataset
        self._models = flex_model  # Add when models are finished
        self._dr_rate = dropout_rate  # Connection dropout rate

    def filter(self, func: Callable):
        """Function that filter the PoolManager by actors giving a function.

        Note: This function doesn't send a copy of the original pool, it sends a reference.
            Changes made on the new pool may affect the original pool.
        Args:
            func (Callable): Function to filter the pool by.

        Returns:
            FlexPoolManger: New filtered pool.
        """
        pass

    def validate(self):
        """
        Validar:
        - Los actores con rol de cliente deben aparecer (mismo id) en self._data.
        - Debe haber, al menos, un actor servidor-aggregador. Puede estar representado por dos actores.
        """
        if any(
            actor_id
            for actor_id in self._actors
            if FlexRoleManager.is_client(self._actors[actor_id])
            and actor_id not in self._data.keys()
        ):
            raise ValueError(
                "All clients must have data. There are some actors with the client Role that doens't have any data."
            )

        if not any(
            actor_id
            for actor_id in self._actors
            if FlexRoleManager.is_server(self._actors[actor_id])
        ):
            raise ValueError("There must be a server in the actor pool.")

        if not any(
            actor_id
            for actor_id in self._actors
            if FlexRoleManager.is_aggregator(self._actors[actor_id])
        ):
            raise ValueError("There must be an aggregator in the actor pool.")

    '''
    La función se implementará cuando se haga el módulo de los modelos.
    def send_model(self, pool: object = None):
        """
        pool es la pool que manda a self el modelo.
        Indicar:
        - pool: pool de la que viene el modelo
        """
        pass

    def deploy_model(self):
        """
        Inicialmente se debe hacer el deploy_model para instanciar el número de modelos que se quiera y no todos.
        Aquí se define el inicio de la ronda para cada usuario en base al modelo que desee entrenar.
        """
        pass

    def aggregation_step(self):
        """
        Función que realizará la agregación de los modelos de los clientes.
        """
        pass
    '''

    @classmethod
    def client_server_architecture(
        cls, fed_dataset: FlexDataset, dropout_rate: float = None
    ):
        actors = FlexActors({actor_id: FlexRole.client for actor_id in fed_dataset.keys()})
        actors["server"] = FlexRole.server_aggregator
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
        return FlexActors(
            {actor_id: FlexRole.server_aggregator_client for actor_id in actors_ids}
        )
