import threading
import unittest
from time import sleep

import pytest

from flex.data.dataset import Dataset
from flex.distributed import ClientBuilder, Server
from flex.model.model import FlexModel
from flex.pool.decorators import collect_clients_weights, set_aggregated_weights

addr, port = "localhost", 8080


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


class TestServer(unittest.TestCase):
    @staticmethod
    def run_client():
        model = FlexModel()
        model["model"] = {"weights1": [1, 2, 3], "weights2": [4, 5, 6]}

        dataset = Dataset.from_array([[1, 2], [3, 4]], [0, 1])

        client = (
            ClientBuilder()
            .collect_weights(collect_weights)
            .model(model)
            .set_weights(set_weights)
            .train(train)
            .eval(eval, dataset)
            .dataset(dataset)
            .build()
        )
        client.run(f"{addr}:{port}")

    def test_server_can_run_collect_ids(self):
        server = Server()
        server.run(addr, port)

        client = threading.Thread(target=self.run_client)
        client.start()

        while len(server) == 0:
            sleep(1)

        ids = server.get_ids()
        self.assertEqual(len(ids), 1)
        server.stop()
        client.join()

        self.assertTrue(len(ids) > 0)
