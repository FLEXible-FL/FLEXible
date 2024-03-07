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

import pytest

from flex.data import Dataset, FedDataDistribution, FedDatasetConfig
from flex.pool.aggregators import fed_avg
from flex.pool.pool import FlexPool
from flex.pool.primitives_tf import (
    collect_clients_weights_tf,
    deploy_server_model_tf,
    evaluate_model_tf,
    init_server_model_tf,
    set_aggregated_weights_tf,
    train_tf,
)


class TestFlexPoolPrimitives(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def _fixture_ag_news_dataset(self):
        from datasets import load_dataset

        config = FedDatasetConfig(
            seed=0,
            n_nodes=2,
            replacement=False,
            node_ids=["client_0", "client_1"],
        )
        self.config = config
        train_data, test_data = load_dataset("imdb", split=["train", "test"])
        self.test_data = test_data
        X_columns = ["text"]
        label_columns = ["label"]
        self.f_imdb = FedDataDistribution.from_config_with_huggingface_dataset(
            train_data, self.config, X_columns, label_columns
        )
        self.f_imdb["server"] = Dataset.from_huggingface_dataset(
            self.test_data, X_columns, label_columns
        )

    def test_primitives_tf(self):
        import numpy as np
        import tensorflow as tf
        import tensorflow_hub as hub

        def define_model(*args):
            model = "https://tfhub.dev/google/nnlm-en-dim50/2"
            hub_layer = hub.KerasLayer(
                model, input_shape=[], dtype=tf.string, trainable=True
            )
            model = tf.keras.Sequential()
            model.add(hub_layer)
            model.add(tf.keras.layers.Dense(16, activation="relu"))
            model.add(tf.keras.layers.Dense(1))
            model.compile(
                optimizer="adam",
                loss=tf.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name="accuracy")],
            )
            return model

        p = FlexPool.client_server_pool(
            self.f_imdb, init_func=init_server_model_tf, model=define_model()
        )
        reference_model = define_model()
        reference_model_params = reference_model.get_weights()
        p.servers.map(deploy_server_model_tf, p.clients)
        assert all(
            np.all(
                np.equal(
                    p._models[k]["model"].get_weights()[0], reference_model_params[0]
                )
            )
            for k in p.actor_ids
        )
        # Train the model
        p.clients.map(train_tf, batch_size=512, epochs=1)
        # Collect weights
        p.aggregators.map(collect_clients_weights_tf, p.clients)
        # Aggregate weights
        p.aggregators.map(fed_avg)
        assert np.any(
            np.not_equal(
                p._models["server"]["aggregated_weights"][0], reference_model_params[0]
            )
        )
        # Transfer weights from aggregators to servers
        p.aggregators.map(set_aggregated_weights_tf, p.servers)
        # Deploy new model to clients
        p.servers.map(deploy_server_model_tf, p.clients)
        reference_value = p._models["server"]["model"].get_weights()
        assert all(
            np.all(np.equal(p._models[k]["model"].get_weights()[0], reference_value[0]))
            for k in p.actor_ids
        )
        result = p.servers.map(evaluate_model_tf)
        print(result)
