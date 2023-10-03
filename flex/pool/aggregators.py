# Copyright 2023 Instituto Andaluz Interuniversitario en Ciencia de Datos e Inteligencia Computacional (DaSCI)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""File that contains the adapted aggregators in FLEXible for fast
development of a federated model in FLEXible.

This aggregators also can work as examples for creating a custom aggregator.
"""

import tensorly as tl

from flex.pool.decorators import aggregate_weights


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
    # Default backend
    # tl.set_backend('numpy')
    for modulename in supported_modules:
        try:
            tmp_import = __import__(modulename)
            if all(
                tmp_import.is_tensor(t) for t in flatten(aggregated_weights_as_list)
            ):
                if modulename == "torch":
                    modulename = f"py{modulename}"
                tl.set_backend(modulename)
        except ImportError:
            ...


@aggregate_weights
def fed_avg(aggregated_weights_as_list: list, *args, **kwargs):
    """Function that implements de FedAvg aggregation method

    Args:
        aggregated_weights_as_list (list): List which contains
        all the weights to aggregate

    Returns:
        np.array: An array with the aggregated weights

    Example of use assuming you are using a client-server architechture:

        from flex.pool.primitive_functions import fed_avg

        aggregator = flex_pool.aggregators
        server = flex_pool.servers
        aggregator.map(server, fed_avg)

    Example of use using the FlexPool without separating server
    and aggregator, and following a client-server architechture.

        from flex.pool.primitive_functions import fed_avg

        flex_pool.aggregators.map(flex_pool.servers, fed_avg)
    """
    set_tensorly_backend(aggregated_weights_as_list)
    agg_weights = []
    for layer_index in range(len(aggregated_weights_as_list[0])):
        weights_per_layer = [
            weights[layer_index] for weights in aggregated_weights_as_list
        ]
        weights_per_layer = tl.stack(weights_per_layer)
        agg_layer = tl.mean(weights_per_layer, axis=0)
        agg_weights.append(agg_layer)
    return agg_weights
