import copy
import unittest

import pytest
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier

from flex.data import FlexDataDistribution
from flex.data.flex_dataset import FlexDataObject
from flex.pool.flex_decorators import (
    aggregate_weights,
    collect_clients_weights,
    deploy_server_model,
    init_server_model,
    set_aggregated_weights,
)
from flex.pool.flex_model import FlexModel
from flex.pool.flex_pool import FlexPool


class TestFlexPool(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_iris_dataset(self):
        iris = load_iris()
        c_iris = FlexDataObject(X_data=iris.data, y_data=iris.target)
        self.f_iris = FlexDataDistribution.iid_distribution(c_iris, n_clients=2)

    def test_decorators(self):
        @init_server_model()
        def build_server_model(
            additional_value=2,
        ):  # These args are store in the server_flex_model
            flex_model = FlexModel()
            flex_model["model"] = KNeighborsClassifier(n_neighbors=3)
            flex_model["additional_value"] = additional_value
            return flex_model

        @deploy_server_model()
        def copy_server_model_to_clients(server_flex_model):
            flex_model = FlexModel()
            for k, v in server_flex_model.items():
                flex_model[k] = copy.deepcopy(v)

            return flex_model

        @collect_clients_weights()
        def get_clients_weights(client_flex_model):
            return client_flex_model["model"].get_params()

        @aggregate_weights()
        def aggregate(list_of_weights):
            return sum(w["leaf_size"] for w in list_of_weights)

        @set_aggregated_weights()
        def set_agreggated_weights_to_server(server_flex_model, aggregated_weights):
            server_flex_model["model"].set_params(leaf_size=aggregated_weights)

        p = FlexPool.client_server_architecture(
            self.f_iris, init_func=build_server_model
        )
        reference_model_params = KNeighborsClassifier(n_neighbors=3).get_params()
        reference_value = reference_model_params["leaf_size"] * len(p.clients)

        p.servers.map(copy_server_model_to_clients, p.clients)
        assert all(
            p._models[k]["model"].get_params() == reference_model_params
            for k in p.actor_ids
        )
        p.aggregators.map(get_clients_weights, p.clients)
        # Aggregate weights
        p.aggregators.map(aggregate)
        assert p._models["server"]["aggregated_weights"] == reference_value
        # Transfer weights from aggregators to servers
        p.aggregators.map(set_agreggated_weights_to_server, p.servers)
        # Deploy new model to clients
        p.servers.map(copy_server_model_to_clients, p.clients)
        assert all(
            p._models[k]["model"].get_params()["leaf_size"] == reference_value
            for k in p.actor_ids
        )
