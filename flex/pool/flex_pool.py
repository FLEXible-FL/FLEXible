from __future__ import annotations

import functools
from collections import defaultdict
from typing import Callable

import numpy as np

from flex.data import FlexDataset
from flex.pool.actors import FlexActors, FlexRole, FlexRoleManager


class FlexPool:
    """
    Class that orchest the training phase of a federated learning experiment.
    The FlexPool class is responsible for orchestating the clients to train a
    federated model.
    This class represents a pool of actors and is in charge of checking the
    communications between them during the process of training a federated model.

    Note: At the moment this class only supports Horizontal Federated Learning,
    but in the future it will cover Vertical Federated Learning and Transfer Learning,
    so users can simulate all the experiments correctly.

    Attributes
    ----------
        - flex_data (FlexDataset): The federated dataset prepared to be used.
        - flex_actors (FlexActors): Actors with its roles.
        - flex_models (defaultdict): A dictionary containing the each actor id,
        and initialized to None. The model to train by each actor will be initialized
        using the map_procedure function following the communication constraints.


    --------------------------------------------------------------------------
    We offer two class methods to create two architectures, client-server architecture
    and p2p architecture. In the client-server architecture, every id from the
    FlexDataset is assigned to be a client, and we create a third-party actor,
    supposed to be neutral, to orchestate the training. Meanwhile, in the p2p
    architecture, each id from the FlexDataset will be assigned to be client,
    server and aggregator. In both cases, the method will create the actors
    so the user will only have to apply the map_procedure function to train the model.

    If the user wants to use a different architecture, she will need to create
    the actors by using the FlexActors class. For example, we let the user create
    a client-server architecture with multiple aggregators, to carry out the aggregation.
    """

    def __init__(
        self,
        flex_data: FlexDataset,
        flex_actors: FlexActors,
        flex_models: defaultdict = None,
    ) -> None:
        self._actors = flex_actors  # Actors
        self._data = flex_data  # FlexDataset
        self._models = flex_models
        if self._models is None:
            self._models = defaultdict(None, {k: None for k in self._actors.keys()})
        self.validate()  # check if the provided arguments generate a valid object

    @classmethod
    def check_compatibility(cls, src_pool, dst_pool):
        """Method to check the compatibility between two different pools.
        This method is used by the map_procedure function to check if the
        function to apply from the source pool to the destination pool can be done.

        Args:
            src_pool (FlexPool): Source pool. Pool that will send the message.
            dst_pool (FlexPool): Destination pool. Pool that will recieve the message.

        Returns:
            bool: True if pools are compatible. False in other case.
        """
        return all(
            FlexRoleManager.check_compatibility(src, dst)
            for _, src in src_pool._actors.items()
            for _, dst in dst_pool._actors.items()
        )

    def map_procedure(self, func: Callable, dst_pool: FlexPool, *args, **kwargs):
        """Method used to send messages from one pool to another. The pool using
        this method is the source pool, and it will send a message, apply a function,
        to the destination pool. The pool sends a message in order to complete a round
        in the Federated Learning (FL) paradigm, so, some examples of the messages
        that will be used by the pools are:
        - send_model: Aggregators send the model to the server when the aggregation is done.
        - aggregation_step: Clients send the model to the aggregator so it can apply the
        aggregation mechanism given.
        - deploy_model: Server sends the global model to the clients once the weights has
        been aggregated.
        - init_model: Server sends the model to train during the learning phase, so the
        clients can initialize it. This is a particular case from the deploy_model case.

        Args:
            func (Callable): Message to pass from the source pool (self) to the destinity pool (dst_pool)
            dst_pool (FlexPool): Pool that will recieve the message from the source pool (self).

        Raises:
            ValueError: This method raises and error if the pools aren't allowed to comunicate

        Returns:
            _type_: The result of applying the function (func) from the source pool (self) to the
            destinty pool (dst_pool).
        """
        if FlexPool.check_compatibility(self, dst_pool):
            return func(self._models, dst_pool._models, *args, **kwargs)
        else:
            raise ValueError(
                "Source and destination pools are not allowed to comunicate, ensure that their actors can communicate."
            )

    def filter(
        self, func: Callable = None, clients_dropout: float = 0.0, *args, **kwargs
    ):
        """Function that filter the PoolManager by actors given a function.
        The function must return True/False, and it recieves the args and kwargs arguments
        for its internal uses. Also, the function recieves an actor_id and an actor_role.
        The actor_id is a string, and the actor_role is a FlexRole.

        Note: This function doesn't send a copy of the original pool, it sends a reference.
            Changes made on the new pool may affect the original pool.
        Args:
            func (Callable): Function to filter the pool by. The function must return True/False.
            clients_dropout (float): Percentage of clients to drop from the training phase. This param
            must be a value in the range [0, 1]. If the clients_dropout > 1, it will return all the
            pool without any changes. For negative values the funcion raises an error.
        Returns:
            FlexPool: New filtered pool.
        """
        if func is None:
            raise ValueError(
                "Function to filter can't be None. Please, provide a function."
            )
        if clients_dropout < 0:
            raise ValueError(
                "The clients dropout can't be negative. Please check use a value in the range [0, 1]"
            )
        clients_dropout = max(1 - min(clients_dropout, 1), 0)
        clients_dropout = int(len(self._actors) * clients_dropout)
        training_clients = np.random.choice(
            list(self._actors.keys()), clients_dropout, replace=False
        )
        new_actors = FlexActors()
        new_data = FlexDataset()
        new_models = defaultdict()
        for actor_id in training_clients:
            if func(actor_id, self._actors[actor_id], *args, **kwargs):
                new_actors[actor_id] = self._actors[actor_id]
                new_models[actor_id] = self._models.get(actor_id)
                if actor_id in self._data:
                    new_data[actor_id] = self._data[actor_id]
        return FlexPool(
            flex_actors=new_actors, flex_data=new_data, flex_models=new_models
        )

    def __len__(self):
        return len(self._actors)

    @functools.cached_property
    def clients(self):
        """Property to get all the clients available in a pool.

        Returns:
            FlexPool: Pool containing all the clients from a pool
        """
        return self.filter(
            lambda a, b: FlexRoleManager.is_client(b), clients_dropout=0.0
        )

    @functools.cached_property
    def aggregators(self):
        """Property to get all the aggregator available in a pool.

        Returns:
            FlexPool: Pool containing all the aggregators from a pool
        """
        return self.filter(lambda a, b: FlexRoleManager.is_aggregator(b))

    @functools.cached_property
    def servers(self):
        """Property to get all the servers available in a pool.

        Returns:
            FlexPool: Pool containing all the servers from a pool
        """
        return self.filter(lambda a, b: FlexRoleManager.is_server(b))

    def validate(self):
        """Function that checks whether the object is correct or not."""
        actors_ids = self._actors.keys()
        data_ids = self._data.keys()
        models_ids = self._models.keys()
        if not (actors_ids >= data_ids and actors_ids >= models_ids):
            raise ValueError("Each node with data or model must have a role asigned")
        for actor_id in self._actors:
            if (
                FlexRoleManager.is_client(self._actors[actor_id])
                and actor_id not in data_ids
            ):
                raise ValueError(
                    "All node with client role must have data. Node with client role and id {actor_id} does not have any data."
                )
        if not (self._actors.keys() <= self._models.keys()):
            raise ValueError(
                "flex_models must have the same keys as flex_actors, but with None value if no model is required."
            )

    @classmethod
    def client_server_architecture(cls, fed_dataset: FlexDataset):
        """Method to create a client-server architeture for a FlexDataset given.
        This functions is used when you have a FlexDataset and you want to start
        the learning phase following a traditional client-server architecture.

        This method will assing to each id from the FlexDataset the client-role,
        and will create a new actor that will be the server-aggregator that will
        orchestrate the learning phase.

        Args:
            fed_dataset (FlexDataset): Federated dataset used to train a model.

        Returns:
            FlexPool: A FlexPool with the assigned roles for a client-server architecture.
        """
        actors = FlexActors(
            {actor_id: FlexRole.client for actor_id in fed_dataset.keys()}
        )
        actors[f"server_{id(cls)}"] = FlexRole.server_aggregator
        return cls(
            flex_data=fed_dataset,
            flex_actors=actors,
            flex_models=None,
        )

    @classmethod
    def p2p_architecture(cls, fed_dataset: FlexDataset):
        """Method to create a peer-to-peer (p2p) architecture for a FlexDataset given.
        This method is used when you have a FlexDataset and you want to start the
        learning phase following a p2p architecture.

        This method will assing all roles (client-aggregator-server) to every id from
        the FlexDataset, so each participant in the learning phase can act as client,
        aggregator and server.

        Args:
            fed_dataset (FlexDataset): Federated dataset used to train a model.

        Returns:
            FlexPool: A FlexPool with the assigned roles for a p2p architecture.
        """
        return cls(
            flex_data=fed_dataset,
            flex_actors=cls.__create_actors_all_privileges(fed_dataset.keys()),
            flex_models=None,
        )

    @classmethod
    def __create_actors_all_privileges(cls, actors_ids):
        """Method that initialize the actors for the pool with all the privileges
        available. This method must be used only when creating a p2p-architecture.

        Args:
            actors_ids (str): The IDs that identify every actor in the pool.

        Returns:
            FlexActors: All the actors for the pool with all the privileges.
        """
        return FlexActors(
            {actor_id: FlexRole.server_aggregator_client for actor_id in actors_ids}
        )
