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
from abc import ABC, abstractmethod
from queue import Queue
from typing import List

import grpc
import numpy as np
from proto.transport_pb2 import ClientMessage, Error, ServerMessage
from proto.transport_pb2_grpc import FlexibleStub

from flex.data import Dataset
from flex.distributed.common import toNumpyArray, toTensorList
from flex.model import FlexModel


class Client(ABC):
    def __init__(self, dataset: Dataset, model: FlexModel):
        self._channel = None
        self._stub = None
        self._q = Queue()
        self._q.put(ClientMessage(handshake_ins=ClientMessage.HandshakeIns(status=200)))
        self.dataset = dataset
        self.model = model

    @staticmethod
    def _iter_queue(q: Queue):
        while True:
            yield q.get()

    def _handle_get_weights_ins(self, response: ServerMessage.GetWeightsIns):
        weights = self.get_weights(self.model)
        for w in weights:
            w = np.array(w, dtype=np.float32)

        self._q.put(
            ClientMessage(
                get_weights_res=ClientMessage.GetWeightsRes(
                    weights=toTensorList(weights)
                )
            )
        )

    def _handle_send_weights_ins(self, response: ServerMessage.SendWeightsIns):
        weights = toNumpyArray(response.weights)
        self.set_weights(self.model, weights)
        self._q.put(
            ClientMessage(send_weights_res=ClientMessage.SendWeightsRes(status=200))
        )

    @abstractmethod
    def set_weights(self, model: FlexModel, weights: List[np.ndarray]):
        pass

    @abstractmethod
    def get_weights(self, model: FlexModel) -> List[np.ndarray]:
        pass

    def _handle_train_ins(self, response: ServerMessage.TrainIns):
        metrics = self.train(self.model, self.dataset)
        if metrics is None or not isinstance(metrics, dict):
            metrics = {}

        self._q.put(ClientMessage(train_res=ClientMessage.TrainRes(metrics=metrics)))

    @abstractmethod
    def train(self, model: FlexModel, data: Dataset):
        pass

    def _handle_eval_ins(self, response: ServerMessage.EvalIns):
        metrics = self.eval(self.model, self.dataset)
        if metrics is None or not isinstance(metrics, dict):
            metrics = {}

        self._q.put(ClientMessage(eval_res=ClientMessage.EvalRes(metrics=metrics)))

    @abstractmethod
    def eval(self, model: FlexModel, data: Dataset):
        pass

    def run(self, address: str):
        self._channel = grpc.insecure_channel(address)
        self._stub = FlexibleStub(self._channel)
        try:
            for response in self._stub.Send(self._iter_queue(self._q)):
                assert isinstance(response, ServerMessage)
                msg = response.WhichOneof("msg")
                if msg == "get_weights_ins":
                    self._handle_get_weights_ins(response.get_weights_ins)
                elif msg == "send_weights_ins":
                    self._handle_send_weights_ins(response.send_weights_ins)
                elif msg == "train_ins":
                    self._handle_train_ins(response.train_ins)
                elif msg == "eval_ins":
                    self._handle_eval_ins(response.eval_ins)
                else:
                    raise Exception("Not implemented")
        except Exception as e:
            print(f"Stopping client: {e}")
            self._q.put(ClientMessage(error=Error(reason="disconected")))
            self._channel.close()
