import threading
import unittest
from time import sleep
import numpy as np
from flex.data.dataset import Dataset
from flex.distributed import ClientBuilder, Server
from flex.model.model import FlexModel
from flex.pool.decorators import collect_clients_weights, set_aggregated_weights

addr, port = "localhost", 8081


@collect_clients_weights
def collect_weights(model):
    return [model["model"]["weights1"]]


@set_aggregated_weights
def set_weights(model, weights):
    model["model"]["weights1"] = weights[0]


def train_fn(model, data):
    return {"loss": 0.1}


def eval_fn(model, data):
    return {"accuracy": 0.5}


class TestGracefulShutdown(unittest.TestCase):
    def setUp(self):
        self.server = None

    def tearDown(self):
        if self.server is not None:
            try:
                self.server.stop()
            except Exception:
                pass  # Server may already be stopped

    def run_client(self, results):
        try:
            model = FlexModel()
            model["model"] = {"weights1": [1, 2, 3]}
            dataset = Dataset.from_array(np.array([[1, 2]]), np.array([0]))

            client = (
                ClientBuilder()
                .collect_weights(collect_weights)
                .model(model)
                .set_weights(set_weights)
                .dataset(dataset)
                .eval(eval_fn, dataset)
                .train(train_fn)
                .build()
            )
            client.run(f"{addr}:{port}")
            results["success"] = True
        except Exception as e:
            results["error"] = str(e)
            results["success"] = False

    def test_graceful_shutdown(self):
        self.server = Server()
        self.server.run(addr, port)

        results = {"success": False}
        client_thread = threading.Thread(target=self.run_client, args=(results,))
        client_thread.start()

        sleep(2)
        self.assertEqual(len(self.server), 1)

        # Stop the server. This should send StopIns and the client should exit gracefully.
        self.server.stop()

        client_thread.join(timeout=10)
        self.assertFalse(client_thread.is_alive(), "Client thread did not exit")
        self.assertTrue(
            results.get("success"), f"Client failed with error: {results.get('error')}"
        )


if __name__ == "__main__":
    unittest.main()
