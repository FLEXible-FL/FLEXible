from __future__ import annotations

import functools
from collections import defaultdict
from typing import Callable

from flex.data import FlexDataset
from flex.pool.actors import FlexActors, FlexRole, FlexRoleManager


class FlexPool:
    """
    Class that orchest the training phase of a federated learning experiment.
    """

    def __init__(
        self,
        flex_data: FlexDataset,
        flex_actors: FlexActors,
        flex_models: defaultdict = None,
        dropout_rate: float = None,
    ) -> None:
        self._actors = flex_actors  # Actors
        self._data = flex_data  # FlexDataset
        self._models = flex_models  # Add when models are finished
        self._dr_rate = dropout_rate  # Connection dropout rate
        self.validate()  # check if the provided arguments generate a valid object

    @classmethod
    def check_compatibility(cls, src_pool, dst_pool):
        return all(
            FlexRoleManager.check_compatibility(src, dst)
            for _, src in src_pool._actors.items()
            for _, dst in dst_pool._actors.items()
        )

    def map_procedure(self, func: Callable, dst_pool: FlexPool, *args, **kwargs):
        if FlexPool.check_compatibility(self, dst_pool):
            return func(self._models, dst_pool._models, *args, **kwargs)
        else:
            raise ValueError(
                "Source and destination pools are not allowed to comunicate, ensure that their actors can communicate."
            )

    def filter(self, func: Callable = None, *args, **kwargs):
        """Function that filter the PoolManager by actors giving a function.
        The function has to return True/False, and it recieves the args and kwargs arguments
        for its internal uses. Also, the function recieves an actor_id and an actor_role.
        The actor_id is a string, and the actor_role is a FlexRole.
        Note: This function doesn't send a copy of the original pool, it sends a reference.
            Changes made on the new pool may affect the original pool.
        Args:
            func (Callable): Function to filter the pool by. The function must return True/False.

        Returns:
            FlexPool: New filtered pool.
        """
        if func is None:
            raise ValueError(
                "Function to filter can't be None. Please, provide a function."
            )
        new_actors = FlexActors()
        new_data = FlexDataset()
        for actor_id in self._actors:
            if func(actor_id, self._actors[actor_id], *args, **kwargs):
                new_actors[actor_id] = self._actors[actor_id]
                if actor_id in self._data:
                    new_data[actor_id] = self._data[actor_id]
                # TODO: Add Model when Model module is finished.
        new_models = defaultdict()
        return FlexPool(
            flex_actors=new_actors, flex_data=new_data, flex_models=new_models
        )

    @functools.cached_property
    def clients(self):
        return self.filter(lambda a, b: FlexRoleManager.is_client(b))

    @functools.cached_property
    def aggregators(self):
        return self.filter(lambda a, b: FlexRoleManager.is_aggregator(b))

    @functools.cached_property
    def servers(self):
        return self.filter(lambda a, b: FlexRoleManager.is_server(b))

    def validate(self):
        """Function that checks whether the object is correct or not."""
        for actor_id in self._actors:
            if (
                FlexRoleManager.is_client(self._actors[actor_id])
                and actor_id not in self._data.keys()
            ):
                raise ValueError(
                    "All node with client role must have data. Node with client role and id {actor_id} does not have any data."
                )

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
        actors = FlexActors(
            {actor_id: FlexRole.client for actor_id in fed_dataset.keys()}
        )
        actors[f"server_{id(cls)}"] = FlexRole.server_aggregator
        return cls(
            flex_data=fed_dataset,
            flex_actors=actors,
            flex_models=defaultdict(),
            dropout_rate=dropout_rate,
        )

    @classmethod
    def p2p_architecture(cls, fed_dataset: FlexDataset, dropout_rate: float = None):
        return cls(
            flex_data=fed_dataset,
            flex_actors=cls.__create_actors_all_privileges(fed_dataset.keys()),
            flex_models=defaultdict(),
            dropout_rate=dropout_rate,
        )

    @classmethod
    def __create_actors_all_privileges(cls, actors_ids):
        return FlexActors(
            {actor_id: FlexRole.server_aggregator_client for actor_id in actors_ids}
        )
