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
import signal
import sys
import threading
from abc import ABC, abstractmethod
from queue import Empty, Queue
from typing import List, Optional

import grpc
import numpy as np

from flex.data import Dataset
from flex.distributed.common import toNumpyArray, toTensorList
from flex.distributed.proto.transport_pb2 import ClientMessage, ServerMessage
from flex.distributed.proto.transport_pb2_grpc import FlexibleStub
from flex.model import FlexModel

logger = logging.getLogger(__name__)


class Client(ABC):
    """
    Allows the implementation for a client in a distributed `FLEXible` environment.
    The abstract methods must be implemented since they teach the client how to retrieve and set weights,
    run training and evaluation. The __init__ method must be called by the child class.
    """

    def __init__(self, dataset: Dataset, model: FlexModel, eval_dataset: Dataset):
        """
        Initializes a Client object.

        Args:
        ----
            dataset (Dataset): The dataset used for training.
            model (FlexModel): The model used for training.
            eval_dataset (Dataset): The dataset used for evaluation.
        """
        self._channel = None
        self._stub = None
        self._q = Queue()
        self._q.put(ClientMessage(handshake_res=ClientMessage.HandshakeRes(status=200)))
        self._finished = threading.Event()
        self.dataset = dataset
        self.model = model
        self.eval_dataset = eval_dataset

    def _iter_queue(self, q: Queue):
        # The iterator may still run if it has still messages to send
        while not self._finished.is_set() or not q.empty():
            try:
                value = q.get(timeout=0.1)
                yield value
            except Empty:
                # This condition is done in order to check finished
                continue
        raise StopIteration

    def _handle_get_weights_ins(self, response: ServerMessage.GetWeightsIns):
        logger.info("Weights requested")
        weights = self.get_weights(self.model)

        self._q.put(
            ClientMessage(
                get_weights_res=ClientMessage.GetWeightsRes(
                    weights=toTensorList([np.array(w) for w in weights])
                )
            )
        )
        logger.info("Weights sent")

    def _handle_send_weights_ins(self, response: ServerMessage.SendWeightsIns):
        logger.info("Weights received")
        weights = toNumpyArray(response.weights)
        self.set_weights(self.model, weights)
        self._q.put(
            ClientMessage(send_weights_res=ClientMessage.SendWeightsRes(status=200))
        )
        logger.info("Weights set")

    def _handle_health_ping(self, response: ServerMessage.HealthPing):
        logger.info("Health ping from server")
        self._q.put(ClientMessage(health_ins=ClientMessage.HealthPing(status=200)))

    @abstractmethod
    def set_weights(self, model: FlexModel, weights: List[np.ndarray]):
        """
        Sets the weights of the given model.

        Args:
        ----
            model (FlexModel): The model to set the weights for.
            weights (List[np.ndarray]): The weights to set for the model.
        """
        pass

    @abstractmethod
    def get_weights(self, model: FlexModel) -> List[np.ndarray]:
        """
        Retrieves the weights of a given FlexModel.

        Args:
        ----
            model (FlexModel): The FlexModel object for which to retrieve the weights.

        Returns:
        -------
            List[np.ndarray]: A list of NumPy arrays representing the weights of the model.
        """
        pass

    def _handle_train_ins(self, response: ServerMessage.TrainIns):
        logger.info("Training requested")
        metrics = self.train(self.model, self.dataset)
        if metrics is None or not isinstance(metrics, dict):
            metrics = {}

        self._q.put(ClientMessage(train_res=ClientMessage.TrainRes(metrics=metrics)))
        logger.info("Training completed")

    @abstractmethod
    def train(self, model: FlexModel, data: Dataset) -> dict:
        """
        Trains the given model using the provided dataset.

        Args:
        ----
            model (FlexModel): The model to be trained.
            data (Dataset): The dataset used for training.

        Returns:
        -------
            dict: A dictionary containing training metrics.
        """
        pass

    def _handle_eval_ins(self, response: ServerMessage.EvalIns):
        logger.info("Evaluation requested")
        metrics = self.eval(self.model, self.eval_dataset)
        if metrics is None or not isinstance(metrics, dict):
            metrics = {}

        self._q.put(ClientMessage(eval_res=ClientMessage.EvalRes(metrics=metrics)))
        logger.info("Evaluation completed")

    @abstractmethod
    def eval(self, model: FlexModel, data: Dataset) -> dict:
        """
        Evaluates the given model on the provided data.

        Args:
        ----
            model (FlexModel): The model to evaluate.
            data (Dataset): The data to evaluate the model on.

        Returns:
        -------
            dict: A dictionary containing evaluation metrics.
        """
        pass

    def run(
        self,
        address: str,
        root_certificate: str = None,
        _stub: Optional[FlexibleStub] = None,
    ):
        if _stub is not None:
            self._stub = _stub
        else:
            if root_certificate is not None:
                self._channel = grpc.secure_channel(
                    target=address,
                    credentials=grpc.ssl_channel_credentials(root_certificate),
                )
            else:
                self._channel = grpc.insecure_channel(address)

            self._stub = FlexibleStub(self._channel)

        self._request_generator = self._stub.Send(self._iter_queue(self._q))
        self._set_exit_handler(self._request_generator)
        try:
            for response in self._request_generator:
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
                elif msg == "health_ins":
                    self._handle_health_ping(response.health_ins)
                else:
                    raise Exception("Not implemented")
        except Exception as e:
            cancelled = False
            if hasattr(e, "details") and callable(e.details):
                cancelled = "Cancelling all calls" in e.details()
            if cancelled:
                logger.info("Disconnected from server")
            else:
                logger.error(f"Canceling process. Got error: {e}")
            self._request_generator.cancel()
        finally:
            self._finished.set()

    def _set_exit_handler(self, request_generator):
        def _handler_(signum, frame):
            request_generator.cancel()
            sys.exit(0)

        # Only set the signal handler if the current thread is the main thread, avoids exceptions on running tests where
        # this code executes in another thread
        if threading.current_thread().__class__.__name__ == "_MainThread":
            signal.signal(signal.SIGINT, _handler_)

    def __del__(self):
        if self._channel is not None:
            self._channel.close()
