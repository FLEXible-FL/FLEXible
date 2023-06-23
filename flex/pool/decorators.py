import functools
import inspect
from typing import List

from flex.model import FlexModel


def __inspect_arguments(func):
    signature = inspect.signature(func)
    try:
        assert len(signature.parameters) >= 1
    except AssertionError as er:
        raise AssertionError(
            f"The decorated function is expected to have at least one argument. {er}"
        ) from er


def init_server_model(func):
    __inspect_arguments(func=func)

    @functools.wraps(func)
    def _init_server_model_(server_flex_model: FlexModel, _, *args, **kwargs):
        server_flex_model.update(func(*args, **kwargs))

    return _init_server_model_


def deploy_server_model(func):
    __inspect_arguments(func=func)

    @functools.wraps(func)
    def _deploy_model_(
        server_flex_model: FlexModel,
        clients_flex_models: List[FlexModel],
        *args,
        **kwargs,
    ):
        for k in clients_flex_models:
            # Reminder, it is not possible to make assignements here
            clients_flex_models[k].update(func(server_flex_model, *args, **kwargs))

    return _deploy_model_


def collect_clients_weights(func):
    __inspect_arguments(func=func)

    @functools.wraps(func)
    def _collect_weights_(
        aggregator_flex_model: FlexModel,
        clients_flex_models: List[FlexModel],
        *args,
        **kwargs,
    ):
        if "weights" not in aggregator_flex_model:
            aggregator_flex_model["weights"] = []
        for k in clients_flex_models:
            client_weights = func(clients_flex_models[k], *args, **kwargs)
            aggregator_flex_model["weights"].append(client_weights)

    return _collect_weights_


def aggregate_weights(func):
    __inspect_arguments(func=func)

    @functools.wraps(func)
    def _aggregate_weights_(aggregator_flex_model: FlexModel, _, *args, **kwargs):
        aggregator_flex_model["aggregated_weights"] = func(
            aggregator_flex_model["weights"], *args, **kwargs
        )
        aggregator_flex_model["weights"] = []

    return _aggregate_weights_


def set_aggregated_weights(func):
    __inspect_arguments(func=func)

    @functools.wraps(func)
    def _deploy_aggregated_weights_(
        aggregator_flex_model: FlexModel,
        servers_flex_models: FlexModel,
        *args,
        **kwargs,
    ):
        for k in servers_flex_models:
            func(
                servers_flex_models[k],
                aggregator_flex_model["aggregated_weights"],
                *args,
                **kwargs,
            )

    return _deploy_aggregated_weights_


def evaluate_server_model(func):
    __inspect_arguments(func=func)

    @functools.wraps(func)
    def _evaluate_server_model_(server_flex_model: FlexModel, _, *args, **kwargs):
        return func(server_flex_model, *args, **kwargs)

    return _evaluate_server_model_
