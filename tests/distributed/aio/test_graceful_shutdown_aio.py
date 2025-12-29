import asyncio
import unittest
import numpy as np
from flex.data.dataset import Dataset
from flex.distributed import ClientBuilder
from flex.distributed.aio import Server
from flex.model.model import FlexModel
from flex.pool.decorators import collect_clients_weights, set_aggregated_weights

addr, port = "localhost", 8082


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


class TestAioGracefulShutdown(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.server = None

    async def asyncTearDown(self):
        if self.server is not None:
            try:
                await self.server.stop()
            except Exception:
                pass  # Server may already be stopped

    async def run_client(self, results):
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
            # Run in a thread because Client.run is blocking
            await asyncio.to_thread(client.run, f"{addr}:{port}")
            results["success"] = True
        except Exception as e:
            results["error"] = str(e)
            results["success"] = False

    async def test_aio_graceful_shutdown(self):
        self.server = Server()
        await self.server.run(addr, port)

        results = {"success": False}
        client_task = asyncio.create_task(self.run_client(results))

        await asyncio.sleep(2)
        self.assertEqual(len(self.server), 1)

        # Stop the server. This should send StopIns and the client should exit gracefully.
        await self.server.stop()

        await asyncio.wait_for(client_task, timeout=10)
        self.assertTrue(
            results.get("success"), f"Client failed with error: {results.get('error')}"
        )


if __name__ == "__main__":
    unittest.main()
