import random
import unittest

import pytest
import tensorly as tl

from flex.pool.aggregators import fed_avg


def simulate_clients_weights_for_module(n_clients, modulename):
    framework = __import__(modulename)
    n_clients = 5
    simulated_client_weights = []
    num_layers = 5
    num_dim = [1, 2, 3, 4, 5]
    layer_ndims = random.sample(num_dim, k=num_layers)
    for _ in range(n_clients):
        random.seed(0)
        simulated_weights = []
        for ndims in layer_ndims:
            tmp_dims = random.sample(
                num_dim, k=ndims
            )  # ndims dimensions of num_dim sizes
            simulated_weights.append(framework.ones(tmp_dims))
        simulated_client_weights.append(simulated_weights)
    return {"weights": simulated_client_weights}


class TestFlexAggregators(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_weights(self):
        self._torch_weights = simulate_clients_weights_for_module(
            n_clients=5, modulename="torch"
        )
        self._tf_weights = simulate_clients_weights_for_module(
            n_clients=5, modulename="tensorflow"
        )
        self._np_weights = simulate_clients_weights_for_module(
            n_clients=5, modulename="numpy"
        )

    def test_fed_avg_with_torch(self):
        fed_avg(self._torch_weights, None)
        agg_weights = self._torch_weights["aggregated_weights"]
        assert tl.get_backend() == "pytorch"
        assert all(
            tl.all(w == tl.ones(tl.shape(w), **tl.context(w))) for w in agg_weights
        )

    def test_fed_avg_with_tf(self):
        fed_avg(self._tf_weights, None)
        agg_weights = self._tf_weights["aggregated_weights"]
        assert tl.get_backend() == "tensorflow"
        assert all(
            tl.all(w == tl.ones(tl.shape(w), **tl.context(w))) for w in agg_weights
        )

    def test_fed_avg_with_np(self):
        fed_avg(self._np_weights, None)
        agg_weights = self._np_weights["aggregated_weights"]
        assert tl.get_backend() == "numpy"
        assert all(
            tl.all(w == tl.ones(tl.shape(w), **tl.context(w))) for w in agg_weights
        )