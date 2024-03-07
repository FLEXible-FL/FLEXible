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
    set_aggregated_weights,
)


@deploy_server_model
def deploy_server_model_pt(server_flex_model, *args, **kwargs):
    """Creates a copy of the server_flex_model and it is set to client nodes using the decorator @deploy_server_model.

    Args:
    -----
        server_flex_model (FlexModel): object storing information needed to run a Pytorch model

    """
    return deepcopy(server_flex_model)


def check_ignored_weights_pt(name, ignore_weights=None):
    """Checks wether name contains any of the words in ignore_weights.

    Args:
    -----
        name (str): name to check
        ignore_weights (list, optional): A list of str. Defaults to None.

    Returns:
    --------
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
    -----
        client_flex_model (FlexModel): A client's FlexModel
        ignore_weights (list): the name of the weights not to collect, by default,
        those containind the words `num_batches_tracked` are not collected, as they
        only make sense in the local model

    Returns:
    --------
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
    import torch

    ignore_weights = kwargs.get("ignore_weights", None)
    with torch.no_grad():
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
                parameters.append(torch.tensor([]))
                continue
            weight_diff = weight_dict[name] - previous_weight_dict[name]
            parameters.append(weight_diff)
    return parameters


@collect_clients_weights
def collect_clients_weights_pt(client_flex_model, *args, **kwargs):
    """Function that collect the weights for a PyTorch model.

    This function returns all the weights of the model.

    Args:
    -----
        client_flex_model (FlexModel): A client's FlexModel
        ignore_weights (list): the name of the weights not to collect, by default,
        those containind the words `num_batches_tracked` are not collected, as they
        only make sense in the local model

    Returns:
    --------
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
    import torch

    ignore_weights = kwargs.get("ignore_weights", None)
    with torch.no_grad():
        parameters = []
        weight_dict = client_flex_model["model"].state_dict()
        for name in weight_dict:
            w = weight_dict[name]
            if check_ignored_weights_pt(name, ignore_weights=ignore_weights):
                w = torch.tensor([])
                continue
            parameters.append(w)
    return parameters


@set_aggregated_weights
def set_aggregated_weights_pt(server_flex_model, aggregated_weights, *args, **kwargs):
    """Function that replaces the weights of the server with the aggregated weights of the aggregator.

    Args:
    -----
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
        weight_dict = server_flex_model["model"].state_dict()
        for layer_key, new in zip(weight_dict, aggregated_weights):
            try:
                if len(new) != 0:  # Do not copy empty layers
                    weight_dict[layer_key].copy_(new)
            except TypeError:  # new has no len property
                weight_dict[layer_key].copy_(new)


@set_aggregated_weights
def set_aggregated_diff_weights_pt(
    server_flex_model, aggregated_diff_weights, *args, **kwargs
):
    """Function to add the aggregated weights to the server.

    Args:
    -----
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
        weight_dict = server_flex_model["model"].state_dict()
        for layer_key, new in zip(weight_dict, aggregated_diff_weights):
            try:
                if len(new) != 0:  # Do not copy empty layers
                    weight_dict[layer_key].add_(new)
            except TypeError:  # new has no len property
                weight_dict[layer_key].add_(new)
