from concurrent import futures
from queue import Queue
from threading import Thread
from typing import Iterator, List, Optional

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
        id,
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
            self.register_queue.put(str(self.id))


class ClientManager:
    def __init__(self, register_queue: Queue):
        self._register_queue = register_queue
        self._clients: List[ClientProxy] = []

    def __len__(self):
        return len(self._clients)

    def get_ids(self) -> List[any]:
        return [client.id for client in self._clients]

    def delete_client(self, client_id):
        self._clients = [
            client for client in self._clients if str(client.id) != str(client_id)
        ]

    # blocking process
    def run_registration(self):
        i = 0
        while True:
            message = self._register_queue.get()
            if isinstance(message, str):
                self.delete_client(message)
                continue

            self._clients.append(
                ClientProxy(
                    i,
                    *message,
                    register_queue=self._register_queue,
                )
            )

    def broadcast(self, message: ServerMessage, node_ids: Optional[List[any]] = None):
        for client in self._clients:
            if node_ids is None or client.id in node_ids:
                client.put_message(message)

    def pool_clients(self, node_ids: Optional[List[any]] = None):
        if node_ids:
            messages = [
                (client.pool_messages(), client.id)
                for client in self._clients
                if client.id in node_ids
            ]
        else:
            messages = [(client.pool_messages(), client.id) for client in self._clients]

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

    def run(self, address: str = "[::]:50051"):
        self._server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        add_FlexibleServicer_to_server(self._servicer, self._server)
        self._server.add_insecure_port(address=address)
        self._server.start()
        Thread(target=self._manager.run_registration, daemon=True).start()
        Thread(target=self._server.wait_for_termination, daemon=True).start()

    def stop(self):
        if self._server is not None:
            self._server.stop(None)


if __name__ == "__main__":
    from time import sleep

    server = Server()
    server.run()
    while len(server) == 0:
        print("Waiting for clients...")
        sleep(1)

    ids = server.get_ids()
    print(f"Ids={ids}")
    server.send_weights(np.ones((1, 1)), node_ids=ids)
    train_metrics = server.train(node_ids=ids)
    print(f"Train metrics = {train_metrics}")
    client_weights = server.collect_weights(node_ids=ids)
    print(f"Weights collected = {client_weights}")
    server.stop()
