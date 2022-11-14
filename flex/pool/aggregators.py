"""File that contains the adapted aggregators in FLEXible for fast
development of a federated model in FLEXible.

This aggregators also can work as examples for creating a custom aggregator.
"""

from flex.pool.decorators import aggregate_weights


@aggregate_weights
def fed_avg(aggregated_weights_as_list, *args, **kwargs):
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
    import numpy as np

    return np.mean(np.array(aggregated_weights_as_list), axis=0)
