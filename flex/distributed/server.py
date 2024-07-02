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
from concurrent import futures
from queue import Queue
from threading import Event, Thread
from typing import Dict, Iterator, List, Optional

import grpc
import numpy as np
from common import toNumpyArray, toTensorList
from grpc import RpcContext

from flex.distributed.proto.transport_pb2 import ClientMessage, ServerMessage
from flex.distributed.proto.transport_pb2_grpc import (
    FlexibleServicer,
    add_FlexibleServicer_to_server,
)


class ClientProxy:
    def __init__(
        self,
        id: str,
        request_iterator: Iterator[ClientMessage],
        communication_queue: Queue,
        register_queue: Queue,
    ):
        self.id = id
        self.request_iterator = request_iterator
        self.communication_queue = communication_queue
        self.register_queue = register_queue

    def __repr__(self):
        return f"ClientProxy({self.id})"

    def put_message(self, message: ServerMessage):
        self.communication_queue.put(message)

    def pool_messages(self):
        try:
            response = next(self.request_iterator)
            if response.WhichOneof("msg") == "error":
                raise StopIteration("Client disconnected")
            return response
        except StopIteration:
            self.register_queue.put(self.id)


class ClientManager:
    def __init__(self, register_queue: Queue):
        self._register_queue = register_queue
        self._clients: Dict[str, ClientProxy] = {}

    def __len__(self):
        return len(self._clients)

    def get_ids(self) -> List[any]:
        return list(self._clients.keys())

    def delete_client(self, client_id):
        if client_id in self._clients:
            del self._clients[client_id]

    # blocking process
    def run_registration(self):
        i = 0
        while True:
            message = self._register_queue.get()
            if isinstance(message, str):
                self.delete_client(message)
                continue

            self._clients[str(i)] = ClientProxy(
                str(i),
                *message,
                register_queue=self._register_queue,
            )

    def broadcast(self, message: ServerMessage, node_ids: Optional[List[str]] = None):
        for id, client in self._clients.items():
            if node_ids is None or id in node_ids:
                client.put_message(message)

    def pool_clients(self, node_ids: Optional[List[str]] = None):
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

        messages = [(m, id) for m, id in messages if m is not None]
        return messages


class ServerServicer(FlexibleServicer):
    def __init__(self, queue: Queue):
        super().__init__()
        self._q = queue

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
        self._q.put((request_iterator, communication_queue))

        while True:
            try:
                yield communication_queue.get()
            except Exception:
                context.cancel()
                break


class Server:
    def __init__(self):
        self._q = Queue()
        self._servicer = ServerServicer(queue=self._q)
        self._manager = ClientManager(register_queue=self._q)
        self._server = None

    def __len__(self):
        return len(self._manager)

    def get_ids(self) -> List[any]:
        return self._manager.get_ids()

    def collect_weights(self, node_ids: Optional[any] = None):
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
        for w in weights:
            w = np.array(w, dtype=np.float32)

        self._manager.broadcast(
            ServerMessage(
                send_weights_ins=ServerMessage.SendWeightsIns(
                    weights=toTensorList(weights)
                )
            ),
            node_ids=node_ids,
        )

        messages = self._manager.pool_clients(node_ids=node_ids)
        assert all(m.WhichOneof("msg") == "send_weights_res" for m, _ in messages)

    def train(self, node_ids: Optional[any] = None):
        self._manager.broadcast(
            ServerMessage(train_ins=ServerMessage.TrainIns(status=200)),
            node_ids=node_ids,
        )
        messages = self._manager.pool_clients(node_ids=node_ids)
        assert all(m.WhichOneof("msg") == "train_res" for m, _ in messages)
        return {id: m.train_res.metrics for m, id in messages}

    def eval(self, node_ids: Optional[any] = None):
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
        address_port = f"{address}:{port}"
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
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
        Thread(target=self._manager.run_registration, daemon=True).start()
        Thread(target=self._server.wait_for_termination, daemon=True).start()

    def stop(self):
        if self._server is not None:
            self._server.stop(None)
