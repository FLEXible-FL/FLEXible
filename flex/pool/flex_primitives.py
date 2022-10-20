"""File that contains the primitive functions to build an easy training loop of the federated learning model.

In this file we specify some functions for each framework, i.e., TensorFlow (tf), PyTorch (pt), among others, but
we only give functions for a general purpose. For a more personalized use of FLEXible, the user must create
her own functions. The user can use this functions as template on how to create a custom funcion for each step
of the training steps in a federated learning environment.

Note that each function is using the decorators we've created to facilitate the use of the library. For a better
understanding on how the platform works, please go to fthe flex_decorators file.
"""
from copy import deepcopy

from flex.pool.flex_decorators import (
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
    from flex.pool.flex_model import FlexModel

    server_flex_model = FlexModel()

    if model._is_compiled:
        server_flex_model["optimizer"] = deepcopy(model.optimizer)
        server_flex_model["loss"] = deepcopy(model.loss)
        server_flex_model["metrics"] = deepcopy(model.compiled_metrics._metrics)
        server_flex_model["model"] = model
    else:
        if any([optimizer, loss, metrics] is None):
            raise ValueError(
                "If the model is not compiled, then optimizer, loss and metrics can't be None. Please, provide a value for this arguments."
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
    """Function the deploy a TensorFlow model from the server to a client.

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

    from flex.pool.flex_model import FlexModel

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
        client_flex_model (FlexModel): client's FlexModel
        client_data (FlexDataset): client's FlexDataset

    Example of use assuming you are using a client-server architechture:

        from flex.pool.primitive_functions import train_tf

        clients = flex_pool.clients

        clients.map(train_tf)

    Example of use using the FlexPool without separating clients
    and following a client-server architechture.

        from flex.pool.primitive_functions import train_tf

        flex_pool.clients.map(train_tf)
    """
    client_flex_model["model"].fit(
        client_data.X_data, client_data.y_data, *args, **kwargs
    )


# def train(client_model, client_data, *args, **kwargs):
#     # TODO: Fulfill this function to work better for PyTorch.
#     # TODO: Improve documentation.
#     """Function to train train a model at client level. This
#     function will use the model allocated at the client level,
#     and then will feed the client's data to the model in orther
#     to train it.

#     We use the args and kwargs arguments to give it to the train
#     function of the model. Each model will have it's own train
#     function, i.e., a TensorFlow model will call .fit() method,
#     so we will call the method as follows:
#     model.fit(X_data, y_data, *args, **kwargs)

#     Args:
#         client_model (_type_): _description_
#         client_data (_type_): _description_
#     """
#     if "verbose" in kwargs and kwargs["verbose"] == 1:
#         print("Training model at client.")

#     if "model" in kwargs or "weights" in kwargs:
#         raise ValueError(
#             "'model' and 'weights' are reserved words in out framework, please, dont't use it."
#         )

#     model = client_model["model"]
#     X_data = client_data.X_data
#     y_data = client_data.y_data
#     if "func" in kwargs:
#         # PyTorch models need it's own function
#         func = kwargs["func"]
#         del kwargs["func"]
#         for item in client_model:
#             if item not in ["model", "weights"]:
#                 kwargs.append(item)
#         func(model, X_data, y_data, *args, **kwargs)

#         def wrapper(func, model, client_data, *args, **kwargs):
#             func(model, client_data, args, kwargs)

#         return wrapper
#     else:
#         model.fit(X_data, y_data, *args, **kwargs)


@collect_clients_weights
def collect_clients_weights_tf(client_flex_model, *args, **kwargs):
    """Function that collect the weights for a TensorFlow model.

    This function returns all the weights of the model.

    Args:
        client_flex_model (FlexModel): A client's FlexModel

    Returns:
        np.array: An array with all the weights of the client's model

    Example of use assuming you are using a client-server architechture:

        from flex.pool.primitive_functions import collect_weights_tf

        clients = flex_pool.clients
        aggregator = flex_pool.aggregators

        clients.map(collect_weights_tf, aggregator)

    Example of use using the FlexPool without separating clients
    and aggregator, and following a client-server architechture.

        from flex.pool.primitive_functions import collect_weights_tf

        flex_pool.clients.map(collect_weights_tf, flex_pool.aggregators)
    """
    return client_flex_model["model"].get_weights()


@collect_clients_weights
def collect_clients_weights_pt(client_flex_mode, *args, **kwargs):
    """Function that collect the weights for a PyTorch model.

    This function returns all the weights of the model.

    Args:
        client_flex_model (FlexModel): A client's FlexModel

    Returns:
        List: List with all the weights of the client's model

    Example of use assuming you are using a client-server architechture:

        from flex.pool.primitive_functions import collect_weights_pt

        clients = flex_pool.clients
        aggregator = flex_pool.aggregators

        clients.map(collect_weights_pt, aggregator)

    Example of use using the FlexPool without separating clients
    and aggregator, and following a client-server architechture.

        from flex.pool.primitive_functions import collect_weights_pt

        flex_pool.clients.map(collect_weights_pt, flex_pool.aggregators)
    """
    return [
        param.cpu().data.numpy() for param in client_flex_mode["model"].parameters()
    ]


@set_aggregated_weights
def set_aggregated_weights_tf(server_flex_model, aggregated_weights, *args, **kwargs):
    """Function that set the aggregated weights by the aggregator to
    the server.

    Args:
        server_flex_model (FlexModel): The server's FlexModel
        aggregated_weights (np.array): An array with the aggregated
        weights of the models.
    """
    server_flex_model["model"].set_weights(aggregated_weights)


@set_aggregated_weights
def set_aggregated_weights_pt(server_flex_model, aggregated_weights, *args, **kwargs):
    """Function to set the aggregated weights to the server

    Args:
        server_flex_model (FlexModel): The server's FlexModel
        aggregated_weights (np.array): Aggregated weights

    Example of use assuming you are using a client-server architechture:

        from flex.pool.primitive_functions import set_aggregated_weights_pt

        aggregator = flex_pool.aggregators

        aggregator.map(set_aggregated_weights_pt)

    Example of use using the FlexPool without separating clients
    and aggregator, and following a client-server architechture.

        from flex.pool.primitive_functions import set_aggregated_weights_pt

        flex_pool.aggregators.map(set_aggregated_weights_pt)
    """
    import torch

    with torch.no_grad():
        for old, new in zip(
            server_flex_model["model"].parameters(), aggregated_weights
        ):
            old.data = torch.from_numpy(new).float()


@evaluate_server_model
def evaluate_server_model_tf(
    server_flex_model, test_data, test_labels, *args, **kwargs
):
    """Function that evaluate the global model on the test data

    Args:
        server_flex_model (FlexModel): server's Flexmodel
        test_data (np.array): Test inputs.
        test_labels (np.array): Test labels.

    Returns:
        Evaluations by the model on the test data.
    """
    return server_flex_model["model"].evaluate(test_data, test_labels)
