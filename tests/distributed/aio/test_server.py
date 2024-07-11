import asyncio
import threading
import unittest

import numpy as np
import pytest

from flex.data.dataset import Dataset
from flex.distributed import ClientBuilder
from flex.distributed.aio.server import Server
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


class TestAsyncServer(unittest.IsolatedAsyncioTestCase):
    @pytest.mark.asyncio
    async def test_server_integration_test(self):
        server = Server()
        await server.run(addr, port)

        client = threading.Thread(target=run_client)
        client.start()
        await asyncio.wait_for(server.wait_for_clients(1), timeout=5)

        ids = server.get_ids()
        self.assertEqual(ids, ["0"], "Client ID is not correct")
        # Ping
        await asyncio.wait_for(server.ping(node_ids=ids), timeout=5)

        # Collect weights from the client
        weights = await server.collect_weights(node_ids=ids)
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
        await server.send_weights(weights=new_weights, node_ids=ids)
        weights = await server.collect_weights(node_ids=ids)
        self.assertTrue(
            all([np.array_equal(w, x) for w, x in zip(weights[0], new_weights)])
        )

        # train
        metrics = await server.train(node_ids=ids)
        self.assertIsInstance(metrics, dict)
        self.assertTrue("0" in metrics)
        self.assertTrue("loss" in metrics["0"])
        self.assertAlmostEqual(metrics["0"]["loss"], 0.1)
        # Check that train has changed weights
        weights = await server.collect_weights(node_ids=ids)
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
        metrics = await server.eval(node_ids=ids)
        self.assertIsInstance(metrics, dict)
        self.assertTrue("0" in metrics)
        self.assertTrue("accuracy" in metrics["0"])
        self.assertAlmostEqual(metrics["0"]["accuracy"], 0.5)

        await server.stop()
        client.join()
