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
import asyncio
import logging
from typing import Dict, Iterator, List, Optional

import grpc
import numpy as np
from grpc import RpcContext

from flex.distributed.common import toNumpyArray, toTensorList
from flex.distributed.proto.transport_pb2 import ClientMessage, ServerMessage
from flex.distributed.proto.transport_pb2_grpc import (
    FlexibleServicer,
    add_FlexibleServicer_to_server,
)

logger = logging.getLogger(__name__)


class ClientProxy:
    """
    Abstraction for working with a client connection.
    Encapsulates the GRPC request iterator providing both methods for sending and pooling messages.

    Args:
    ----
        id (str): The ID of the client.
        request_iterator (Iterator[ClientMessage]): The iterator for receiving client messages.
        communication_queue (Queue): The queue for sending messages to the client.
        register_queue (Queue): The queue for registering the client.

    Attributes:
    ----------
        id (str): The ID of the client.
        request_iterator (Iterator[ClientMessage]): The iterator for receiving client messages.
        communication_queue (Queue): The queue for sending messages to the client.
        register_queue (Queue): The queue for registering the client.
    """

    def __init__(
        self,
        id: str,
        request_iterator: Iterator[ClientMessage],
        communication_queue: asyncio.Queue,
    ):
        self.id = id
        self.request_iterator = request_iterator
        self.communication_queue = communication_queue

    def __repr__(self):
        return f"ClientProxy({self.id})"

    async def put_message(self, message: ServerMessage):
        """
        Sends a message to the client.

        Args:
        ----
            message (ServerMessage): The message to be sent to the client.
        """
        logger.info(f"Sending message to {self.id}. Message: {message}")
        await self.communication_queue.put(message)

    async def pool_messages(self):
        """
        Pools messages from the client.

        Returns
        -------
            ClientMessage: The next message received from the client.
        """
        try:
            response = await anext(self.request_iterator)
            if response.WhichOneof("msg") == "error":
                logger.error(
                    f"Client {self.id} sent an error message: {response.error}"
                )
                raise StopAsyncIteration("Client disconnected")
            return response
        except StopAsyncIteration:
            logger.info(f"Client {self.id} disconnected due to no more messages")
            await self.communication_queue.put("Client disconnected")


class ClientManager:
    """
    The ClientManager class manages the collection of ClientProxys for the connected clients.
    It provides methods for registering clients, deleting clients, broadcasting messages to clients,
    and pooling messages from clients.

    When the server starts to run, it spawns two threads. One thread runs the registration method
    of the ClientManager, which listens for new client registrations and adds them to the collection
    of clients. The other thread waits for termination methods and listens for new connections.

    Attributes
    ----------
        _register_queue (Queue): The queue used for registering new clients.
        _clients (Dict[str, ClientProxy]): The collection of ClientProxys for the connected clients.
    """

    def __init__(self, register_queue: asyncio.Queue):
        self._register_queue = register_queue
        self._clients: Dict[str, ClientProxy] = {}

    def __len__(self):
        return len(self._clients)

    def get_ids(self) -> List[str]:
        """
        Returns a list of client IDs currently connected to the server.

        Returns
        -------
            List[str]: A list of client IDs.
        """
        return list(self._clients.keys())

    def delete_client(self, client_id):
        """
        Deletes a client from the collection of clients.

        Args:
        ----
            client_id: The ID of the client to delete.
        """
        if client_id in self._clients:
            del self._clients[client_id]

    async def run_registration(self):
        """
        The registration method that runs in a separate thread.
        Listens for new client registrations and adds them to the collection of clients.
        """
        i = 0
        while True:
            message = await self._register_queue.get()
            request_iterator, communication_queue = message
            self._clients[str(i)] = ClientProxy(
                str(i), request_iterator, communication_queue
            )
            logger.info(f"Client {i} registered")

    async def broadcast(
        self, message: ServerMessage, node_ids: Optional[List[str]] = None
    ):
        """
        Broadcasts a message to all or a subset of clients.

        Args:
        ----
            message (ServerMessage): The message to broadcast.
            node_ids (Optional[List[str]]): The IDs of the clients to broadcast the message to.
                If None, the message will be broadcasted to all clients.
        """
        if node_ids:
            if not set(node_ids).issubset(self._clients):
                logger.warning("Some node IDs are not registered")

        futures = []
        for id, client in self._clients.items():
            if node_ids is None or id in node_ids:
                futures.append(client.put_message(message))

        await asyncio.gather(*futures)

    async def pool_clients(self, node_ids: Optional[List[str]] = None):
        """
        Pools messages from all or a subset of clients.

        Args:
        ----
            node_ids (Optional[List[str]]): The IDs of the clients to pool messages from.
                If None, messages will be pooled from all clients.

        Returns:
        -------
            List[Tuple[Optional[ClientMessage], str]]: A list of tuples containing the pooled messages
            and their corresponding client IDs.
        """
        if node_ids:
            messages_fut = [
                (client.pool_messages(), id)
                for id, client in self._clients.items()
                if id in node_ids
            ]
        else:
            messages_fut = [
                (client.pool_messages(), id) for id, client in self._clients.items()
            ]

        # wait for all messages to be received
        results = await asyncio.gather(
            *[m for m, _ in messages_fut], return_exceptions=True
        )
        messages = [(result, id) for (_, id), result in zip(messages_fut, results)]

        messages_not_none = [(m, id) for m, id in messages if m is not None]
        for id in [id for m, id in messages if m is None]:
            logger.info(f"Unregistering client {id}")
            self.delete_client(id)

        return messages_not_none


class ServerServicer(FlexibleServicer):
    def __init__(self, queue: asyncio.Queue):
        super().__init__()
        self._registration_queue = queue

    @staticmethod
    def _handshake(message: ClientMessage):
        if (
            message.WhichOneof("msg") != "handshake_res"
            or message.handshake_res.status != 200
        ):
            raise grpc.RpcError("No handshake message")

    async def Send(
        self, request_iterator: Iterator[ClientMessage], context: RpcContext
    ):
        first_request = await anext(request_iterator)
        self._handshake(first_request)
        communication_queue = asyncio.Queue()
        await self._registration_queue.put((request_iterator, communication_queue))
        logger.info("Client connected")

        while True:
            try:
                value = await communication_queue.get()
                if value == "Client disconnected":
                    context.cancel()
                    break
                yield value
            except Exception:
                context.cancel()
                break


class Server:
    """
    Server for distributed FLEXible environment.

    This class represents a server in a distributed FLEXible environment. It provides methods for managing clients,
    collecting weights, sending weights, training models, and evaluating models.
    """

    def __init__(self):
        self._registration_queue = asyncio.Queue()
        self._servicer = ServerServicer(queue=self._registration_queue)
        self._manager = ClientManager(
            register_queue=self._registration_queue,
        )
        self._server = None

    def __len__(self):
        """
        Returns the number of registered clients.

        Returns
        -------
            int: The number of registered clients.
        """
        return len(self._manager)

    def get_ids(self) -> List[str]:
        """
        Returns a list of client IDs.

        Returns
        -------
            List[str]: A list of client IDs.
        """
        return self._manager.get_ids()

    async def ping(self, node_ids: Optional[List[str]] = None):
        """
        Sends a ping message to the specified nodes and retrieves their health status.

        Args:
        ----
            node_ids (Optional[List[str]]): A list of node IDs to send the ping message to. If not provided, the ping message will be sent to all nodes.

        Returns:
        -------
            None
        """
        await self._manager.broadcast(
            ServerMessage(health_ins=ServerMessage.HealthPing(status=200)),
            node_ids=node_ids,
        )
        messages = await self._manager.pool_clients(node_ids=node_ids)
        assert all(m.WhichOneof("msg") == "health_ins" for m, _ in messages)

    async def collect_weights(self, node_ids: Optional[List[str]] = None):
        """
        Collects weights from clients.

        Args:
        ----
            node_ids (Optional[List[str]]): Optional list of client IDs. If provided, only the specified clients will be
                used for weight collection.

        Returns:
        -------
            List[np.ndarray]: A list of collected weights.
        """
        await self._manager.broadcast(
            ServerMessage(get_weights_ins=ServerMessage.GetWeightsIns(status=200)),
            node_ids=node_ids,
        )
        messages = await self._manager.pool_clients(node_ids=node_ids)
        assert all(m.WhichOneof("msg") == "get_weights_res" for m, _ in messages)
        rv = []
        for message, _ in messages:
            weights = toNumpyArray(message.get_weights_res.weights)
            rv.append(weights)

        return rv

    async def send_weights(
        self, weights: List[np.ndarray], node_ids: Optional[List[str]] = None
    ):
        """
        Sends weights to clients.

        Args:
        ----
            weights (List[np.ndarray]): A list of weights to send to clients.
            node_ids (Optional[List[str]]): Optional list of client IDs. If provided, only the specified clients will receive
                the weights.
        """
        await self._manager.broadcast(
            ServerMessage(
                send_weights_ins=ServerMessage.SendWeightsIns(
                    weights=toTensorList([np.array(w) for w in weights])
                )
            ),
            node_ids=node_ids,
        )

        messages = await self._manager.pool_clients(node_ids=node_ids)
        assert all(m.WhichOneof("msg") == "send_weights_res" for m, _ in messages)

    async def train(
        self, node_ids: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Trains models on clients.

        Args:
        ----
            node_ids (Optional[List[str]]): Optional list of client IDs. If provided, only the specified clients will be used
                for training.

        Returns:
        -------
            Dict[str, Dict[str, float]]: A dictionary mapping client IDs to training metrics.
        """
        await self._manager.broadcast(
            ServerMessage(train_ins=ServerMessage.TrainIns(status=200)),
            node_ids=node_ids,
        )
        messages = await self._manager.pool_clients(node_ids=node_ids)
        assert all(m.WhichOneof("msg") == "train_res" for m, _ in messages)
        return {id: m.train_res.metrics for m, id in messages}

    async def eval(
        self, node_ids: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluates models on clients.

        Args:
        ----
            node_ids (Optional[List[str]]): Optional list of client IDs. If provided, only the specified clients will be used
                for evaluation.

        Returns:
        -------
            Dict[str, Dict[str, float]]: A dictionary mapping client IDs to evaluation metrics.
        """
        await self._manager.broadcast(
            ServerMessage(eval_ins=ServerMessage.EvalIns(status=200)),
            node_ids=node_ids,
        )
        messages = await self._manager.pool_clients(node_ids=node_ids)
        assert all(m.WhichOneof("msg") == "eval_res" for m, _ in messages)
        return {id: m.eval_res.metrics for m, id in messages}

    async def wait_for_clients(self, num_clients: int):
        """
        Waits for a specific number of clients to connect.

        Args:
        ----
            num_clients (int): The number of clients to wait for.
        """
        while len(self) < num_clients:
            await asyncio.sleep(1)

    async def run(
        self,
        address: str = "[::]",
        port: int = 50051,
        ssl_private_key: str = None,
        ssl_certificate_chain: str = None,
        ssl_root_certificate: str = None,
        require_client_auth: bool = False,
    ):
        """
        Starts the server. Does not block the main thread.

        Note: This method may be called only once.

        Args:
        ----
            address (str): The address to bind the server to. Defaults to "[::]".
            port (int): The port to bind the server to. Defaults to 50051.
            ssl_private_key (str): The path to the SSL private key file. If provided along with
                `ssl_certificate_chain`, the server will use secure gRPC communication.
            ssl_certificate_chain (str): The path to the SSL certificate chain file. If provided along with
                `ssl_private_key`, the server will use secure gRPC communication.
            ssl_root_certificate (str): The path to the SSL root certificate file. Required if `require_client_auth`
                is set to True.
            require_client_auth (bool): Whether to require client authentication. Defaults to False.
        """
        if self._server is not None:
            raise RuntimeError("Server is already running")

        address_port = f"{address}:{port}"
        self._server = grpc.aio.server()
        add_FlexibleServicer_to_server(self._servicer, self._server)

        if ssl_private_key is not None and ssl_certificate_chain is not None:
            if require_client_auth:
                assert (
                    ssl_root_certificate is not None
                ), "Root certificate must be provided if client authentication is required"
            self._server.add_secure_port(
                address=address_port,
                server_credentials=grpc.ssl_server_credentials(
                    [(ssl_private_key, ssl_certificate_chain)],
                    root_certificates=ssl_root_certificate,
                    require_client_auth=require_client_auth,
                ),
            )
        else:
            self._server.add_insecure_port(address=address_port)

        await self._server.start()
        logger.info(f"Server started at {address_port}")
        # Start the registration and termination threads
        self._termination = asyncio.Task(self._server.wait_for_termination())
        self._registration = asyncio.Task(self._manager.run_registration())

    async def stop(self):
        """
        Stops the server.
        """
        if self._server is not None:
            await self._server.stop(grace=None)
            self._server = None
        self._termination.cancel()
        self._registration.cancel()
