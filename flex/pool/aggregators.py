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
"""File that contains the adapted aggregators in FLEXible for fast
development of a federated model in FLEXible.

This aggregators also can work as examples for creating a custom aggregator.
"""

import tensorly as tl  # noqa: E402

from flex.pool.decorators import aggregate_weights  # noqa: E402


def flatten(xs):
    for x in xs:
        if isinstance(x, (list, tuple)):
            yield from flatten(x)
        else:
            yield x


def set_tensorly_backend(
    aggregated_weights_as_list: list, supported_modules: list = None
):  # jax support is planned
    if supported_modules is None:
        supported_modules = ["tensorflow", "torch"]
    backend_set = False
    for modulename in supported_modules:
        try:
            tmp_import = __import__(modulename)
            if all(
                tmp_import.is_tensor(t) for t in flatten(aggregated_weights_as_list)
            ):
                if modulename == "torch":
                    modulename = f"py{modulename}"
                tl.set_backend(modulename)
                backend_set = True
                break
            else:
                del tmp_import
        except ImportError:
            ...
    # Default backend
    if not backend_set:
        tl.set_backend("numpy")


def fed_avg_f(aggregated_weights_as_list: list):
    n_nodes = len(aggregated_weights_as_list)
    ponderation = [1 / n_nodes] * n_nodes
    return weighted_fed_avg_f(aggregated_weights_as_list, ponderation)


def weighted_fed_avg_f(aggregated_weights_as_list: list, ponderation: list):
    n_layers = len(aggregated_weights_as_list[0])
    agg_weights = []
    for layer_index in range(n_layers):
        weights_per_layer = []
        for client_weights, p in zip(aggregated_weights_as_list, ponderation):
            context = tl.context(client_weights[layer_index])
            w = client_weights[layer_index] * tl.tensor(p, **context)
            weights_per_layer.append(w)
        weights_per_layer = tl.stack(weights_per_layer)
        agg_layer = tl.sum(weights_per_layer, axis=0)
        agg_weights.append(agg_layer)
    return agg_weights


@aggregate_weights
def fed_avg(aggregated_weights_as_list: list):
    """Function that implements the FedAvg aggregation method

    Args:
    -----
        aggregated_weights_as_list (list): List which contains
        all the weights to aggregate

    Returns:
    --------
        tensor array: An array with the aggregated weights

    Example of use assuming you are using a client-server architecture:

        from flex.pool.primitive_functions import fed_avg

        aggregator = flex_pool.aggregators
        server = flex_pool.servers
        aggregator.map(server, fed_avg)

    Example of use using the FlexPool without separating server
    and aggregator, and following a client-server architecture.

        from flex.pool.primitive_functions import fed_avg

        flex_pool.aggregators.map(flex_pool.servers, fed_avg)
    """
    set_tensorly_backend(aggregated_weights_as_list)
    return fed_avg_f(aggregated_weights_as_list)


@aggregate_weights
def weighted_fed_avg(aggregated_weights_as_list: list, ponderation: list):
    """Function that implements the weighted FedAvg aggregation method.

    Args:
    -----
        aggregated_weights_as_list (list): List which contains
        all the weights to aggregate
        ponderation (list): weights assigned to each client

    Returns:
    --------
        tensor array: An array with the aggregated weights

    Example of use assuming you are using a client-server architecture:

        from flex.pool.primitive_functions import weighted_fed_avg

        aggregator = flex_pool.aggregators
        server = flex_pool.servers
        dummy_poderation = [1.]*len(flex_pool.clients)
        aggregator.map(server, weighted_fed_avg, ponderation=dummy_poderation)

    Example of use using the FlexPool without separating server
    and aggregator, and following a client-server architecture.

        from flex.pool.primitive_functions import weighted_fed_avg
        dummy_poderation = [1.]*len(flex_pool.clients)
        flex_pool.aggregators.map(flex_pool.servers, weighted_fed_avg, ponderation=dummy_poderation)
    """
    set_tensorly_backend(aggregated_weights_as_list)
    return weighted_fed_avg_f(aggregated_weights_as_list, ponderation)
