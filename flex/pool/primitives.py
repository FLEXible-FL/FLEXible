"""File that contains the primitive functions to build an easy training loop of the federated learning model.

In this file we specify some functions for each framework, i.e., TensorFlow (tf), PyTorch (pt), among others, but
we only give functions for a general purpose. For a more personalized use of FLEXible, the user must create
her own functions. The user can use this functions as template on how to create a custom function for each step
of the training steps in a federated learning environment.

Note that each function is using the decorators we've created to facilitate the use of the library. For a better
understanding on how the platform works, please go to the flex_decorators file.
"""
from copy import deepcopy

import numpy as np

from flex.pool.decorators import (
    collect_clients_weights,
    deploy_server_model,
    evaluate_server_model,
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
        model (tf.keras.Model): A tf.keras.model initialized.
        optimizer (tf.keras.optimizers, optional): Optimizer for the model. Defaults to None.
        loss (tf.keras.losses, optional): _description_. Defaults to None.
        metrics (tf.keras.metrics, optional): _description_. Defaults to None.

    Raises:
        ValueError: If the model is not compiled and any of the optimizer, loss or metrics
        is not provided, then it will raise an error because we can't initialize
        the model.

    Returns:
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
        server_flex_model (FlexModel): Server FlexModel

    Returns:
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


@deploy_server_model
def deploy_server_model_pt(server_flex_model, *args, **kwargs):
    """Creates a copy of the server_flex_model and it is set to client nodes using the decorator @deploy_server_model.

    Args:
    ----
        server_flex_model (FlexModel): object storing information needed to run a Pytorch model

    """
    return deepcopy(server_flex_model)


def train_tf(client_flex_model, client_data, *args, **kwargs):
    """Function of general purpose to train a TensorFlow model
    using FLEXible.

    Args:
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
        client_flex_model (FlexModel): A client's FlexModel

    Returns:
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


def check_ignored_weights_pt(name, ignore_weights=None):
    """Checks wether name contains any of the words in ignore_weights.

    Args:
    ----
        name (str): name to check
        ignore_weights (list, optional): A list of str. Defaults to None.

    Returns:
    -------
        bool: True if any og the elements of list ignore_weights is present in name, otherwise False.
    """
    if ignore_weights is None:
        ignore_weights = ["num_batches_tracked"]
    return any(ignored in name for ignored in ignore_weights)


@collect_clients_weights
def collect_client_diff_weights_pt(client_flex_model, *args, **kwargs):
    # sourcery skip: raise-specific-error
    """Function that collect the weights for a PyTorch model. Particularly,
        it collects the difference between the model before and after training, \
        that is, what the model has learnt in its local training step. Also note \
        that the weights of the model before training are assume to be stored \
        using `previous_model` as key.

    This function returns the weights of the model.

    Args:
    ----
        client_flex_model (FlexModel): A client's FlexModel
        ignore_weights (list): the name of the weights not to collect, by default,
        those containind the words `num_batches_tracked` are not collected, as they
        only make sense in the local model

    Returns:
    -------
        List: List with the weights of the client's model

    Example of use assuming you are using a client-server architecture:

        from flex.pool import collect_client_diff_weights_pt

        clients = flex_pool.clients
        aggregator = flex_pool.aggregators

        clients.map(collect_client_diff_weights_pt, aggregator)

    Example of using the FlexPool without separating clients
    and aggregator, and following a client-server architecture.

        from flex.pool import collect_client_diff_weights_pt

        flex_pool.clients.map(collect_client_diff_weights_pt, flex_pool.aggregators)
    """
    ignore_weights = kwargs.get("ignore_weights", None)
    weight_dict = client_flex_model["model"].state_dict()
    try:
        previous_weight_dict = client_flex_model["previous_model"].state_dict()
    except KeyError as e:
        raise Exception(
            'A copy of the model before training must be stored in client FlexModel using key: "previous_model"'
        ) from e
    parameters = []
    for name in weight_dict:
        if check_ignored_weights_pt(name, ignore_weights=ignore_weights):
            parameters.append(np.array([]))
            continue
        weight_diff = weight_dict[name].cpu() - previous_weight_dict[name].cpu()
        parameters.append(weight_diff.data.numpy())
    return parameters


@collect_clients_weights
def collect_clients_weights_pt(client_flex_model, *args, **kwargs):
    """Function that collect the weights for a PyTorch model.

    This function returns all the weights of the model.

    Args:
        client_flex_model (FlexModel): A client's FlexModel
        ignore_weights (list): the name of the weights not to collect, by default,
        those containind the words `num_batches_tracked`are not collected, as they
        only make sense in the local model

    Returns:
        List: List with all the weights of the client's model

    Example of use assuming you are using a client-server architecture:

        from flex.pool import collect_weights_pt

        clients = flex_pool.clients
        aggregator = flex_pool.aggregators

        clients.map(collect_weights_pt, aggregator)

    Example of using the FlexPool without separating clients
    and aggregator, and following a client-server architecture.

        from flex.pool import collect_weights_pt

        flex_pool.clients.map(collect_weights_pt, flex_pool.aggregators)
    """

    ignore_weights = kwargs.get("ignore_weights", None)
    parameters = []
    weight_dict = client_flex_model["model"].state_dict()
    for name in weight_dict:
        w = weight_dict[name].cpu().data.numpy()
        if check_ignored_weights_pt(name, ignore_weights=ignore_weights):
            w = np.array([])
            continue
        parameters.append(w)
    return parameters


@set_aggregated_weights
def set_aggregated_weights_tf(server_flex_model, aggregated_weights, *args, **kwargs):
    """Function that replaces the weights of the server with the aggregated weights of the aggregator.

    Args:
        server_flex_model (FlexModel): The server's FlexModel
        aggregated_weights (np.array): An array with the aggregated
        weights of the models.
    """
    server_flex_model["model"].set_weights(aggregated_weights)


@set_aggregated_weights
def set_aggregated_weights_pt(server_flex_model, aggregated_weights, *args, **kwargs):
    """Function that replaces the weights of the server with the aggregated weights of the aggregator.

    Args:
        server_flex_model (FlexModel): The server's FlexModel
        aggregated_weights (np.array): Aggregated weights

    Example of use assuming you are using a client-server architecture:

        from flex.pool import set_aggregated_weights_pt

        aggregator = flex_pool.aggregators

        aggregator.map(set_aggregated_weights_pt)

    Example of using the FlexPool without separating clients
    and aggregator, and following a client-server architecture.

        from flex.pool import set_aggregated_weights_pt

        flex_pool.aggregators.map(set_aggregated_weights_pt)
    """
    import torch

    with torch.no_grad():
        old_weight_dict = server_flex_model["model"].state_dict()
        for old, new in zip(old_weight_dict, aggregated_weights):
            try:
                if len(new) != 0:  # Do not copy empty layers
                    old_weight_dict[old].data = torch.from_numpy(new).float()
            except TypeError:  # new has no len property
                old_weight_dict[old].data = torch.from_numpy(new).float()


@set_aggregated_weights
def set_aggregated_diff_weights_pt(
    server_flex_model, aggregated_diff_weights, *args, **kwargs
):
    """Function to add the aggregated weights to the server.

    Args:
    ----
        server_flex_model (FlexModel): The server's FlexModel
        aggregated_diff_weights (np.array): Aggregated weights

    Example of use assuming you are using a client-server architecture:

        from flex.pool import set_aggregated_diff_weights_pt

        aggregator = flex_pool.aggregators

        aggregator.map(set_aggregated_diff_weights_pt)

    Example of using the FlexPool without separating clients
    and aggregator, and following a client-server architecture.

        from flex.pool import set_aggregated_diff_weights_pt

        flex_pool.aggregators.map(set_aggregated_diff_weights_pt)
    """
    import torch

    with torch.no_grad():
        server_flex_model["model"] = server_flex_model["model"].to("cpu")
        old_weight_dict = server_flex_model["model"].state_dict()
        for old, new in zip(old_weight_dict, aggregated_diff_weights):
            try:
                if len(new) != 0:  # Do not copy empty layers
                    old_weight_dict[old].data += torch.from_numpy(new).float()
            except TypeError:  # new has no len property
                old_weight_dict[old].data += torch.from_numpy(new).float()


@evaluate_server_model
def evaluate_server_model_tf(server_flex_model, test_data, test_labels):
    """Function that evaluate the global model on the test data

    Args:
        server_flex_model (FlexModel): server's FlexModel
        test_data (np.array): Test inputs.
        test_labels (np.array): Test labels.

    Returns:
        Evaluations by the model on the test data.
    """
    return server_flex_model["model"].evaluate(test_data, test_labels, verbose=False)
