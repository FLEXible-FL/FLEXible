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
import logging
from concurrent import futures
from queue import Empty, Queue
from threading import Event, Thread
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
        communication_queue: Queue,
        disconnected_event: Event,
    ):
        self.id = id
        self.request_iterator = request_iterator
        self.communication_queue = communication_queue
        self.disconnected_event = disconnected_event

    def __repr__(self):
        return f"ClientProxy({self.id})"

    def put_message(self, message: ServerMessage):
        """
        Sends a message to the client.

        Args:
        ----
            message (ServerMessage): The message to be sent to the client.
        """
        logger.info(f"Sending message to {self.id}. Message: {message}")
        self.communication_queue.put(message)

    def pool_messages(self):
        """
        Pools messages from the client.

        Returns
        -------
            ClientMessage: The next message received from the client.
        """
        try:
            response = next(self.request_iterator)
            if response.WhichOneof("msg") == "error":
                logger.error(
                    f"Client {self.id} sent an error message: {response.error}"
                )
                raise StopIteration("Client disconnected")
            return response
        except StopIteration:
            logger.info(f"Client {self.id} disconnected due to no more messages")
            self.communication_queue.put("Client disconnected")
            pass

    def disconnect(self):
        self.disconnected_event.set()

    def __del__(self):
        self.disconnected_event.set()


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

    def __init__(self, register_queue: Queue, register_stop_event: Event):
        self._register_queue = register_queue
        self._clients: Dict[str, ClientProxy] = {}
        self._register_stop_event = register_stop_event

    def __len__(self):
        return len(self._clients)

    def get_ids(self) -> List[any]:
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

    def run_registration(self):
        """
        The registration method that runs in a separate thread.
        Listens for new client registrations and adds them to the collection of clients.
        """
        i = 0
        while not self._register_stop_event.is_set():
            try:
                message = self._register_queue.get(timeout=0.1)
                request_iterator, communication_queue, finishing_event = message
                self._clients[str(i)] = ClientProxy(
                    str(i), request_iterator, communication_queue, finishing_event
                )
                logger.info(f"Client {i} registered")
            except Empty:
                pass

        for client in self._clients.values():
            client.disconnect()

    def broadcast(self, message: ServerMessage, node_ids: Optional[List[str]] = None):
        """
        Broadcasts a message to all or a subset of clients.

        Args:
        ----
            message (ServerMessage): The message to broadcast.
            node_ids (Optional[List[str]]): The IDs of the clients to broadcast the message to.
                If None, the message will be broadcasted to all clients.
        """
        for id, client in self._clients.items():
            if node_ids is None or id in node_ids:
                client.put_message(message)

    def pool_clients(self, node_ids: Optional[List[str]] = None):
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
            messages = [
                (client.pool_messages(), id)
                for id, client in self._clients.items()
                if id in node_ids
            ]
        else:
            messages = [
                (client.pool_messages(), id) for id, client in self._clients.items()
            ]

        messages_not_none = [(m, id) for m, id in messages if m is not None]
        for id in [id for m, id in messages if m is None]:
            logger.info(f"Unregistering client {id}")
            self.delete_client(id)

        return messages_not_none


class ServerServicer(FlexibleServicer):
    def __init__(self, queue: Queue):
        super().__init__()
        self._registration_queue = queue

    @staticmethod
    def _handshake(message: ClientMessage):
        if (
            message.WhichOneof("msg") != "handshake_ins"
            or message.handshake_ins.status != 200
        ):
            raise grpc.RpcError("No handshake message")

    def Send(
        self, request_iterator: Iterator[ClientMessage], context: RpcContext
    ) -> Iterator[ServerMessage]:
        first_request = next(request_iterator)
        self._handshake(first_request)
        communication_queue = Queue()
        finishing_event = Event()
        self._registration_queue.put(
            (request_iterator, communication_queue, finishing_event)
        )
        logger.info("Client connected")

        while finishing_event.is_set() is False:
            try:
                value = communication_queue.get(timeout=0.1)
                if value == "Client disconnected":
                    context.cancel()
                    break
                yield value
            except Empty:
                continue
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
        self._registration_queue = Queue()
        self._stop_event = Event()
        self._servicer = ServerServicer(queue=self._registration_queue)
        self._manager = ClientManager(
            register_queue=self._registration_queue,
            register_stop_event=self._stop_event,
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

    def get_ids(self) -> List[any]:
        """
        Returns a list of client IDs.

        Returns
        -------
            List[any]: A list of client IDs.
        """
        return self._manager.get_ids()

    def collect_weights(self, node_ids: Optional[any] = None):
        """
        Collects weights from clients.

        Args:
        ----
            node_ids (Optional[any]): Optional list of client IDs. If provided, only the specified clients will be
                used for weight collection.

        Returns:
        -------
            List[np.ndarray]: A list of collected weights.
        """
        self._manager.broadcast(
            ServerMessage(get_weights_ins=ServerMessage.GetWeightsIns(status=200)),
            node_ids=node_ids,
        )
        messages = self._manager.pool_clients(node_ids=node_ids)
        assert all(m.WhichOneof("msg") == "get_weights_res" for m, _ in messages)
        rv = []
        for message, _ in messages:
            weights = toNumpyArray(message.get_weights_res.weights)
            rv.append(weights)

        return rv

    def send_weights(self, weights: List[np.ndarray], node_ids: Optional[any] = None):
        """
        Sends weights to clients.

        Args:
        ----
            weights (List[np.ndarray]): A list of weights to send to clients.
            node_ids (Optional[any]): Optional list of client IDs. If provided, only the specified clients will receive
                the weights.
        """
        self._manager.broadcast(
            ServerMessage(
                send_weights_ins=ServerMessage.SendWeightsIns(
                    weights=toTensorList([np.array(w) for w in weights])
                )
            ),
            node_ids=node_ids,
        )

        messages = self._manager.pool_clients(node_ids=node_ids)
        assert all(m.WhichOneof("msg") == "send_weights_res" for m, _ in messages)

    def train(self, node_ids: Optional[any] = None):
        """
        Trains models on clients.

        Args:
        ----
            node_ids (Optional[any]): Optional list of client IDs. If provided, only the specified clients will be used
                for training.

        Returns:
        -------
            Dict[any, any]: A dictionary mapping client IDs to training metrics.
        """
        self._manager.broadcast(
            ServerMessage(train_ins=ServerMessage.TrainIns(status=200)),
            node_ids=node_ids,
        )
        messages = self._manager.pool_clients(node_ids=node_ids)
        assert all(m.WhichOneof("msg") == "train_res" for m, _ in messages)
        return {id: m.train_res.metrics for m, id in messages}

    def eval(self, node_ids: Optional[any] = None):
        """
        Evaluates models on clients.

        Args:
        ----
            node_ids (Optional[any]): Optional list of client IDs. If provided, only the specified clients will be used
                for evaluation.

        Returns:
        -------
            Dict[any, any]: A dictionary mapping client IDs to evaluation metrics.
        """
        self._manager.broadcast(
            ServerMessage(eval_ins=ServerMessage.EvalIns(status=200)),
            node_ids=node_ids,
        )
        messages = self._manager.pool_clients(node_ids=node_ids)
        assert all(m.WhichOneof("msg") == "eval_res" for m, _ in messages)
        return {id: m.eval_res.metrics for m, id in messages}

    def run(
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
        address_port = f"{address}:{port}"
        self._executor = futures.ThreadPoolExecutor(max_workers=10)
        self._server = grpc.server(self._executor)
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

        self._server.start()
        logger.info(f"Server started at {address_port}")
        self._registration = Thread(target=self._manager.run_registration, daemon=True)
        self._termination = Thread(
            target=self._server.wait_for_termination, daemon=True
        )
        # Start the registration and termination threads
        self._registration.start()
        self._termination.start()

    def stop(self):
        """
        Stops the server.
        """
        self._stop_event.set()
        if self._server is not None:
            event = self._server.stop(None)
            event.wait()
        self._termination.join()
        self._registration.join()
        self._executor.shutdown(cancel_futures=True, wait=False)
