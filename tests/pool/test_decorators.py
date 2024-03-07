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
import copy
import unittest

import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from flex.data import FedDataDistribution
from flex.data.fed_dataset import Dataset
from flex.model.model import FlexModel
from flex.pool.decorators import (
    aggregate_weights,
    collect_clients_weights,
    deploy_server_model,
    evaluate_server_model,
    init_server_model,
    set_aggregated_weights,
)
from flex.pool.pool import FlexPool


class TestFlexPool(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_iris_dataset(self):
        iris = load_iris()
        c_iris = Dataset.from_array(iris.data, iris.target)
        self.f_iris = FedDataDistribution.iid_distribution(c_iris, n_nodes=2)

    def test_decorators_guard(self):
        with pytest.raises(AssertionError):

            @set_aggregated_weights
            def bad_set_agreggated_weights_to_server(server_flex_model):
                pass

    def test_decorators(self):
        @init_server_model
        def build_server_model():
            flex_model = FlexModel()
            flex_model["model"] = KNeighborsClassifier(n_neighbors=3)
            return flex_model

        @deploy_server_model
        def copy_server_model_to_clients(server_flex_model):
            flex_model = FlexModel()
            for k, v in server_flex_model.items():
                flex_model[k] = copy.deepcopy(v)

            return flex_model

        @collect_clients_weights
        def get_clients_weights(client_flex_model):
            return client_flex_model["model"].get_params()

        @aggregate_weights
        def aggregate(list_of_weights):
            return sum(w["leaf_size"] for w in list_of_weights)

        @set_aggregated_weights
        def set_agreggated_weights_to_server(server_flex_model, aggregated_weights):
            server_flex_model["model"].set_params(leaf_size=aggregated_weights)

        @evaluate_server_model
        def evaluate_server(server_flex_model, test_data=None):
            x, y = test_data.to_numpy()
            server_flex_model["model"].fit(x, y)
            preds = server_flex_model["model"].predict(x)
            return preds, accuracy_score(preds, y)

        p = FlexPool.client_server_pool(self.f_iris, init_func=build_server_model)
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
        test_data = self.f_iris[0]
        result = p.servers.map(evaluate_server, test_data=test_data)
        preds, accuracy = result[0]
        print(accuracy)
        assert len(preds) == len(test_data)
