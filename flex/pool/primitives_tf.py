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

"""File that contains the primitive functions to build an easy training loop of the federated learning model.

In this file we specify some functions for each framework, i.e., TensorFlow (tf), PyTorch (pt), among others, but
we only give functions for a general purpose. For a more personalized use of FLEXible, the user must create
her own functions. The user can use this functions as template on how to create a custom function for each step
of the training steps in a federated learning environment.

Note that each function is using the decorators we've created to facilitate the use of the library. For a better
understanding on how the platform works, please go to the flex_decorators file.
"""
from copy import deepcopy  # noqa: E402

from flex.pool.decorators import (  # noqa: E402
    collect_clients_weights,
    deploy_server_model,
    init_server_model,
    set_aggregated_weights,
)


@init_server_model
def init_server_model_tf(
    model=None, optimizer=None, loss=None, metrics=None, *args, **kwargs
):
    """Function that initialize a model in the server side for the TensorFlow framework.

    This function acts as a message handler, that will initialize
    the model at the server side in a client-server architecture.

    Args:
    -----
        model (tf.keras.Model): A tf.keras.model initialized.
        optimizer (tf.keras.optimizers, optional): Optimizer for the model. Defaults to None.
        loss (tf.keras.losses, optional): _description_. Defaults to None.
        metrics (tf.keras.metrics, optional): _description_. Defaults to None.

    Raises:
    -------
        ValueError: If the model is not compiled and any of the optimizer, loss or metrics
        is not provided, then it will raise an error because we can't initialize
        the model.

    Returns:
    --------
        FlexModel: A FlexModel that will be assigned to the server.
    """
    from flex.model.model import FlexModel

    server_flex_model = FlexModel()

    if model._is_compiled:
        server_flex_model["optimizer"] = deepcopy(model.optimizer)
        server_flex_model["loss"] = deepcopy(model.loss)
        server_flex_model["metrics"] = deepcopy(model.compiled_metrics._metrics)
        server_flex_model["model"] = model
    else:
        if any([optimizer, loss, metrics] is None):
            raise ValueError(
                "If the model is not compiled, then optimizer, loss and metrics can't be None. Please, provide a "
                "value for these arguments. "
            )
        server_flex_model["optimizer"] = optimizer
        server_flex_model["loss"] = loss
        server_flex_model["metrics"] = metrics
        server_flex_model["model"] = model
        # Compile the model
        server_flex_model["model"].compile(
            optimizer=server_flex_model["optimizer"],
            loss=server_flex_model["loss"],
            metrics=server_flex_model["metrics"],
        )
    return server_flex_model


@deploy_server_model
def deploy_server_model_tf(server_flex_model, *args, **kwargs):
    """Function to deploy a TensorFlow model from the server to a client.

    The function will make a deepcopy for a TensorFlow model, as it needs
    a special method of copying. Also, it compiles the model for being able
    to train the model.

    This function uses the decorator @deploy_server_model to deploy the
    server_flex_model to the all the clients, so we only need to create
    the steps for 1 client.

    Args:
    -----
        server_flex_model (FlexModel): Server FlexModel

    Returns:
    --------
        FlexModel: The client's FlexModel
    """
    import tensorflow as tf

    from flex.model.model import FlexModel

    weights = server_flex_model["model"].get_weights()
    model = tf.keras.models.clone_model(server_flex_model["model"])
    model.set_weights(weights)
    model.compile(
        optimizer=server_flex_model["optimizer"],
        loss=server_flex_model["loss"],
        metrics=server_flex_model["metrics"],
    )
    client_flex_model = FlexModel()
    client_flex_model["model"] = model
    return client_flex_model


def train_tf(client_flex_model, client_data, *args, **kwargs):
    """Function of general purpose to train a TensorFlow model
    using FLEXible.

    Args:
    -----
        client_flex_model (FlexModel): client's FlexModel
        client_data (FedDataset): client's FedDataset

    Example of use assuming you are using a client-server architecture:

        from flex.pool import train_tf

        clients = flex_pool.clients

        clients.map(train_tf)

    Example of using the FlexPool without separating clients
    and following a client-server architecture.

        from flex.pool import train_tf

        flex_pool.clients.map(train_tf)
    """
    X, y = client_data.to_numpy()
    client_flex_model["model"].fit(X, y, *args, **kwargs)


@collect_clients_weights
def collect_clients_weights_tf(client_flex_model, *args, **kwargs):
    """Function that collect the weights for a TensorFlow model.

    This function returns all the weights of the model.

    Args:
    -----
        client_flex_model (FlexModel): A client's FlexModel

    Returns:
    --------
        np.array: An array with all the weights of the client's model

    Example of use assuming you are using a client-server architecture:

        from flex.pool import collect_weights_tf

        clients = flex_pool.clients
        aggregator = flex_pool.aggregators

        clients.map(collect_weights_tf, aggregator)

    Example of using the FlexPool without separating clients
    and aggregator, and following a client-server architecture.

        from flex.pool import collect_weights_tf

        flex_pool.clients.map(collect_weights_tf, flex_pool.aggregators)
    """
    return client_flex_model["model"].get_weights()


@set_aggregated_weights
def set_aggregated_weights_tf(server_flex_model, aggregated_weights, *args, **kwargs):
    """Function that replaces the weights of the server with the aggregated weights of the aggregator.

    Args:
    -----
        server_flex_model (FlexModel): The server's FlexModel
        aggregated_weights (np.array): An array with the aggregated
        weights of the models.
    """
    server_flex_model["model"].set_weights(aggregated_weights)


def evaluate_model_tf(flex_model, test_data):
    """Function that evaluate the global model on the test data.

    Args:
    -----
        flex_model (FlexModel): server's FlexModel
        test_data (Dataset): Test inputs.

    Returns:
    --------
        Evaluations by the model on the test data.
    """
    X, y = test_data.to_numpy()
    return flex_model["model"].evaluate(X, y, verbose=False)
