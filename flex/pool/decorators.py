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
import functools
from typing import List

from flex.common.utils import check_min_arguments
from flex.model import FlexModel


def ERROR_MSG_MIN_ARG_GENERATOR(f, min_args):
    return f"The decorated function: {f.__name__} is expected to have at least {min_args} argument/s."


def init_server_model(func):
    @functools.wraps(func)
    def _init_server_model_(server_flex_model: FlexModel, _, *args, **kwargs):
        server_flex_model.update(func(*args, **kwargs))

    return _init_server_model_


def deploy_server_model(func):
    min_args = 1
    assert check_min_arguments(func, min_args), ERROR_MSG_MIN_ARG_GENERATOR(
        func, min_args
    )

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
    min_args = 1
    assert check_min_arguments(func, min_args), ERROR_MSG_MIN_ARG_GENERATOR(
        func, min_args
    )

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
    min_args = 1
    assert check_min_arguments(func, min_args), ERROR_MSG_MIN_ARG_GENERATOR(
        func, min_args
    )

    @functools.wraps(func)
    def _aggregate_weights_(aggregator_flex_model: FlexModel, _, *args, **kwargs):
        aggregator_flex_model["aggregated_weights"] = func(
            aggregator_flex_model["weights"], *args, **kwargs
        )
        aggregator_flex_model["weights"] = []

    return _aggregate_weights_


def set_aggregated_weights(func):
    min_args = 2
    assert check_min_arguments(func, min_args), ERROR_MSG_MIN_ARG_GENERATOR(
        func, min_args
    )

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
    min_args = 1
    assert check_min_arguments(func, min_args), ERROR_MSG_MIN_ARG_GENERATOR(
        func, min_args
    )

    @functools.wraps(func)
    def _evaluate_server_model_(server_flex_model: FlexModel, _, *args, **kwargs):
        return func(server_flex_model, *args, **kwargs)

    return _evaluate_server_model_
