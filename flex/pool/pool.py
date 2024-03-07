"""
Copyright (C) 2024  Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI).

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
from __future__ import annotations

import functools
import random
from typing import Callable, Hashable, Union

from flex.actors.actors import FlexActors, FlexRoleManager
from flex.actors.architectures import client_server_architecture, p2p_architecture
from flex.data import FedDataset
from flex.model.model import FlexModel


class FlexPool:
    """
    Class that orchest the training phase of a federated learning experiment.
    The FlexPool class is responsible for orchestating the nodes to train a
    federated model.
    This class represents a pool of actors and is in charge of checking the
    communications between them during the process of training a federated model.

    Note: At the moment this class only supports Horizontal Federated Learning,
    but in the future it will cover Vertical Federated Learning and Transfer Learning,
    so users can simulate all the experiments correctly.

    Attributes
    ----------
        - flex_data (FedDataset): The federated dataset prepared to be used.
        - flex_actors (FlexActors): Actors with its roles.
        - flex_models (defaultdict): A dictionary containing the each actor id,
        and initialized to None. The model to train by each actor will be initialized
        using the map function following the communication constraints.


    We offer two class methods to create two architectures, client-server architecture
    and p2p architecture. In the client-server architecture, every id from the
    FedDataset is assigned to be a client, and we create a third-party actor,
    supposed to be neutral, to orchestate the training. Meanwhile, in the p2p
    architecture, each id from the FedDataset will be assigned to be client,
    server and aggregator. In both cases, the method will create the actors
    so the user will only have to apply the map function to train the model.

    If the user wants to use a different architecture, she will need to create
    the actors by using the FlexActors class. For example, we let the user create
    a client-server architecture with multiple aggregators, to carry out the aggregation.
    """

    def __init__(
        self,
        flex_data: FedDataset,
        flex_actors: FlexActors,
        flex_models: dict[Hashable, FlexModel] = None,
    ) -> None:
        self._actors = flex_actors  # Actors
        self._data = flex_data  # FedDataset
        self._models = flex_models
        if self._models is None:
            self._models = {k: FlexModel() for k in self._actors}
            for k in self._models:
                self._models[k].actor_id = k
        self.validate()  # check if the provided arguments generate a valid object

    @classmethod
    def check_compatibility(cls, src_pool, dst_pool):
        """Method to check the compatibility between two different pools.
        This method is used by the map function to check if the
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

    def map(self, func: Callable, dst_pool: FlexPool = None, **kwargs):
        r"""Method used to send messages from one pool to another. The pool using
        this method is the source pool, and it will send a message, apply a function,
        to the destination pool. If no destination pool is provided, then the function is applied
        to the source pool. The pool sends a message in order to complete a round
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
        -----
            func (Callable): If dst_pool is None, then message is sent to the source (self). In this situation
            the function func is called for each actor in the pool, providing actor's data and actor's model
            as arguments in addition to \*args and \**kwargs. If dst_pool is not None, the message is sent from
            the source pool (self) to the destination pool (dst_pool). The function func is called for each actor
            in the pool, providing the model of the current actor in the source pool and all the models of the
            actors in the destination pool.

            dst_pool (FlexPool): Pool that will recieve the message from the source pool (self), it can be None.

        Raises:
        -------
            ValueError: This method raises and error if the pools aren't allowed to comunicate

        Returns:
        --------
            List[Any]: A list of the result of applying the function (func) from the source pool (self) to the
            destination pool (dst_pool). If dst_pool is None, then the results come from the source pool. The
            length of the returned values equals the number of actors in the source pool.
        """
        if dst_pool is None:
            res = [
                func(self._models.get(i), self._data.get(i), **kwargs)
                for i in self._actors
            ]
        elif FlexPool.check_compatibility(self, dst_pool):
            res = [
                func(self._models.get(i), dst_pool._models, **kwargs)
                for i in self._actors
            ]
        else:
            raise ValueError(
                "Source and destination pools are not allowed to comunicate, ensure that their actors can communicate."
            )
        if all(ele is not None for ele in res):
            return res

    def select(self, criteria: Union[int, Callable], *criteria_args, **criteria_kwargs):
        """Function that returns a subset of a FlexPool meeting a certain criteria.
        If criteria is an integer, a subset of the available nodes of size criteria is
        randomly sampled. If criteria is a function, then we select those nodes where
        the function returns True values. Note that, the function must have at least
        two arguments, a node id and the roles associated to such node id.
        The actor_id is a string, and the actor_role is a FlexRole object.

        Note: This function doesn't send a copy of the original pool, it sends a reference.
            Changes made on the returned pool affect the original pool.

        Args:
        -----
            criteria (int, Callable): if a function is provided, then it must return
            True/False values for each pair of node_id, node_roles. Additional arguments
            required for the function are passed in criteria_args and criteria_kwargs.
            Otherwise, criteria is interpreted as number of nodes to randomly sample from the pool.
            criteria_args: additional args required for the criteria function. Otherwise ignored.
            criteria_kwargs: additional keyword args required for the criteria function. Otherwise ignored.

        Returns:
        --------
            FlexPool: a pool that contains the nodes that meet the criteria.
        """
        new_actors = FlexActors()
        new_data = FedDataset()
        new_models = {}
        available_nodes = list(self._actors.keys())
        if callable(criteria):
            selected_keys = [
                actor_id
                for actor_id in available_nodes
                if criteria(
                    actor_id, self._actors[actor_id], *criteria_args, **criteria_kwargs
                )
            ]
        else:
            num_nodes = criteria
            selected_keys = random.sample(available_nodes, num_nodes)

        for actor_id in selected_keys:
            new_actors[actor_id] = self._actors[actor_id]
            new_models[actor_id] = self._models[actor_id]
            if actor_id in self._data:
                new_data[actor_id] = self._data[actor_id]

        return FlexPool(
            flex_actors=new_actors, flex_data=new_data, flex_models=new_models
        )

    def __len__(self):
        return len(self._actors)

    @functools.cached_property
    def actor_ids(self):
        return list(self._actors.keys())

    @functools.cached_property
    def clients(self):
        """Property to get all the clients available in a pool.

        Returns:
        --------
            FlexPool: Pool containing all the clients from a pool
        """
        return self.select(lambda a, b: FlexRoleManager.is_client(b))

    @functools.cached_property
    def aggregators(self):
        """Property to get all the aggregator available in a pool.

        Returns:
        --------
            FlexPool: Pool containing all the aggregators from a pool
        """
        return self.select(lambda a, b: FlexRoleManager.is_aggregator(b))

    @functools.cached_property
    def servers(self):
        """Property to get all the servers available in a pool.

        Returns:
        --------
            FlexPool: Pool containing all the servers from a pool
        """
        return self.select(lambda a, b: FlexRoleManager.is_server(b))

    def validate(self):  # sourcery skip: de-morgan
        """Function that checks whether the object is correct or not."""
        actors_ids = self._actors.keys()
        data_ids = self._data.keys()
        models_ids = self._models.keys()
        for actor_id in actors_ids:
            if (
                FlexRoleManager.is_client(self._actors[actor_id])
                and actor_id not in data_ids
            ):
                raise ValueError(
                    "All node with client role must have data. Node with client role and id {actor_id} does not have any data."
                )
            if actor_id not in self._models:
                raise ValueError(
                    f"All nodes must have a FlexModel object associated as a model, but {actor_id} does not."
                )
        flex_models_ids = {self._models[k].actor_id for k in self._models}
        if not (
            actors_ids >= data_ids
            and actors_ids >= models_ids
            and actors_ids >= flex_models_ids
        ):  # noqa: E501
            raise ValueError("Each node with data or model must have a role asigned")

    @classmethod
    def client_server_pool(
        cls,
        fed_dataset: FedDataset,
        init_func: Callable,
        server_id: str = "server",
        **kwargs,
    ):
        """Method to create a client-server architeture for a FlexDataset given.
        This functions is used when you have a FlexDataset and you want to start
        the learning phase following a traditional client-server architecture.

        This method will assing to each id from the FlexDataset the client-role,
        and will create a new actor that will be the server-aggregator that will
        orchestrate the learning phase.

        Args:
        -----
            fed_dataset (FedDataset): Federated dataset used to train a model.

        Returns:
        --------
            FlexPool: A FlexPool with the assigned roles for a client-server architecture.
        """
        actors = client_server_architecture(fed_dataset.keys(), server_id)

        new_arch = cls(
            flex_data=fed_dataset,
            flex_actors=actors,
            flex_models=None,
        )
        new_arch.servers.map(init_func, **kwargs)
        return new_arch

    @classmethod
    def p2p_pool(cls, fed_dataset: FedDataset, init_func: Callable, **kwargs):
        """Method to create a peer-to-peer (p2p) architecture for a FlexDataset given.
        This method is used when you have a FlexDataset and you want to start the
        learning phase following a p2p architecture.

        This method will assing all roles (client-aggregator-server) to every id from
        the FlexDataset, so each participant in the learning phase can act as client,
        aggregator and server.

        Args:
        -----
            fed_dataset (FedDataset): Federated dataset used to train a model.

        Returns:
        --------
            FlexPool: A FlexPool with the assigned roles for a p2p architecture.
        """
        new_arch = cls(
            flex_data=fed_dataset,
            flex_actors=p2p_architecture(fed_dataset),
            flex_models=None,
        )
        new_arch.servers.map(init_func, **kwargs)
        return new_arch

    def __iter__(self):
        yield from self._actors
