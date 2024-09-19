import threading
import unittest
from time import sleep

import numpy as np

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
    model["model"]["weights2"] = [7, 8, 9]
    return {"loss": 0.1}


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

    def test_server_run_called_twice(self):
        server = Server()
        server.run(addr, port)
        with self.assertRaises(RuntimeError):
            server.run(addr, port)

    def test_server_integration_test(self):
        server = Server()
        server.run(addr, port)

        client = threading.Thread(target=self.run_client)
        client.start()
        sleep(1)

        ids = server.get_ids()
        self.assertEqual(ids, ["0"], "Client ID is not correct")
        # Collect weights from the client
        server.ping()
        weights = server.collect_weights(node_ids=ids)
        self.assertTrue(
            all(
                [
                    np.array_equal(w, x)
                    for w, x in zip(
                        weights[0], [np.array([1, 2, 3]), np.array([4, 5, 6])]
                    )
                ]
            )
        )

        # set weights
        new_weights = [np.array([10, 11, 12]), np.array([13, 14, 15])]
        server.send_weights(weights=new_weights, node_ids=ids)
        weights = server.collect_weights(node_ids=ids)
        self.assertTrue(
            all([np.array_equal(w, x) for w, x in zip(weights[0], new_weights)])
        )

        # train
        metrics = server.train(node_ids=ids)
        self.assertIsInstance(metrics, dict)
        self.assertTrue("0" in metrics)
        self.assertTrue("loss" in metrics["0"])
        self.assertAlmostEqual(metrics["0"]["loss"], 0.1)
        # Check that train has changed weights
        weights = server.collect_weights(node_ids=ids)
        self.assertTrue(
            all(
                [
                    np.array_equal(w, x)
                    for w, x in zip(
                        weights[0], [np.array([10, 11, 12]), np.array([7, 8, 9])]
                    )
                ]
            ),
            f"weights: {weights}, expected: {[np.array([10, 11, 12]), np.array([7, 8, 9])]}",
        )

        # Eval
        metrics = server.eval(node_ids=ids)
        self.assertIsInstance(metrics, dict)
        self.assertTrue("0" in metrics)
        self.assertTrue("accuracy" in metrics["0"])
        self.assertAlmostEqual(metrics["0"]["accuracy"], 0.5)

        server.stop()
        client.join()
