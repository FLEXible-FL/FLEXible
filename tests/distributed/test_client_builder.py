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
import unittest
from unittest import mock

import numpy as np
import pytest

from flex.data import Dataset
from flex.distributed import ClientBuilder
from flex.distributed.common import (  # required for internal testing
    toNumpyArray,
    toTensorList,
)
from flex.distributed.proto.transport_pb2 import (  # required for internal testing
    ServerMessage,
)
from flex.model import FlexModel
from flex.pool.decorators import (
    collect_clients_weights,
    init_server_model,
    set_aggregated_weights,
)


@init_server_model
def build_server_model():
    model = FlexModel()
    model["model"] = {"weights1": [1, 2, 3], "weights2": [4, 5, 6]}
    return model


@collect_clients_weights
def collect_weights(model):
    return [model["model"]["weights1"], model["model"]["weights2"]]


@set_aggregated_weights
def set_weights(model, weights):
    model["model"]["weights1"] = weights[0]
    model["model"]["weights2"] = weights[1]


def train(model, data):
    model["model"]["weights1"] = [7, 8, 9]


def eval(model, data):
    return {"accuracy": 0.5}


class TestClientBuilder(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_dataset(self):
        self.dataset = Dataset.from_array([[1, 2], [3, 4]], [0, 1])

    def test_builder_raises_when_not_all_fields_are_called(self):
        with pytest.raises(AssertionError):
            ClientBuilder().collect_weights(collect_weights).set_weights(
                set_weights
            ).train(train).build()

    def test_builder_builds_working_client(self):
        client = (
            ClientBuilder()
            .collect_weights(collect_weights)
            .build_model(build_server_model)
            .set_weights(set_weights)
            .train(train)
            .eval(eval, self.dataset)
            .dataset(self.dataset)
            .build()
        )

        mock_stub = mock.Mock()
        mock_stub.Send = mock.Mock(return_value=mock.MagicMock())
        iter_mock = mock_stub.Send.return_value
        iter_mock.__iter__ = mock.Mock(
            return_value=iter(
                [
                    ServerMessage(
                        get_weights_ins=ServerMessage.GetWeightsIns(status=200)
                    ),
                    ServerMessage(train_ins=ServerMessage.TrainIns(status=200)),
                    ServerMessage(eval_ins=ServerMessage.EvalIns(status=200)),
                    ServerMessage(
                        get_weights_ins=ServerMessage.GetWeightsIns(status=200)
                    ),
                    ServerMessage(
                        send_weights_ins=ServerMessage.SendWeightsIns(
                            weights=toTensorList(np.array([[1, 2], [3, 4]]))
                        )
                    ),
                    ServerMessage(
                        get_weights_ins=ServerMessage.GetWeightsIns(status=200)
                    ),
                ]
            )
        )

        client.run("localhost:50051", _stub=mock_stub)

        # Obtain the generator passed to the stub where the client sends messages
        mock_stub.Send.assert_called_once()
        generator = mock_stub.Send.mock_calls[0].args[0]

        # Always start with handshake
        response = next(generator)
        self.assertEqual(response.handshake_res.status, 200)
        # obtain initial weights
        response = next(generator)
        weights = toNumpyArray(response.get_weights_res.weights)
        self.assertEqual([w.tolist() for w in weights], [[1, 2, 3], [4, 5, 6]])
        # train
        response = next(generator)
        # eval
        response = next(generator)
        self.assertEqual(response.eval_res.metrics["accuracy"], 0.5)
        # see if weights changed
        response = next(generator)
        weights = toNumpyArray(response.get_weights_res.weights)
        self.assertEqual([w.tolist() for w in weights], [[7, 8, 9], [4, 5, 6]])
        # send weights
        response = next(generator)
        self.assertEqual(response.send_weights_res.status, 200)
        # get weights again
        response = next(generator)
        weights = toNumpyArray(response.get_weights_res.weights)
        self.assertEqual([w.tolist() for w in weights], [[1, 2], [3, 4]])

        # The StopIteration in the generator raises a RuntimeError
        self.assertRaises(RuntimeError, lambda: next(generator))
