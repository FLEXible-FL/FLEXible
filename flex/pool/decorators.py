import functools


def init_server_model(func):
    @functools.wraps(func)
    def _init_server_model_(server_flex_model, _, *args, **kwargs):
        server_flex_model.update(func(*args, **kwargs))

    return _init_server_model_


def deploy_server_model(func):
    @functools.wraps(func)
    def _deploy_model_(server_flex_model, clients_flex_models, *args, **kwargs):
        for k in clients_flex_models:
            # Reminder, it is not possible to make assignements here
            clients_flex_models[k].update(func(server_flex_model, *args, **kwargs))

    return _deploy_model_


def collect_clients_weights(func):
    @functools.wraps(func)
    def _collect_weights_(aggregator_flex_model, clients_flex_models, *args, **kwargs):
        if "weights" not in aggregator_flex_model:
            aggregator_flex_model["weights"] = []
        for k in clients_flex_models:
            client_weights = func(clients_flex_models[k], *args, **kwargs)
            aggregator_flex_model["weights"].append(client_weights)

    return _collect_weights_


def aggregate_weights(func):
    @functools.wraps(func)
    def _aggregate_weights_(aggregator_flex_model, _, *args, **kwargs):
        aggregator_flex_model["aggregated_weights"] = func(
            aggregator_flex_model["weights"], *args, **kwargs
        )
        aggregator_flex_model["weights"] = []

    return _aggregate_weights_


def set_aggregated_weights(func):
    @functools.wraps(func)
    def _deploy_aggregated_weights_(
        aggregator_flex_model, servers_flex_models, *args, **kwargs
    ):
        for k in servers_flex_models:
            func(
                servers_flex_models[k],
                aggregator_flex_model["aggregated_weights"],
                *args,
                **kwargs
            )

    return _deploy_aggregated_weights_


def evaluate_server_model(func):
    @functools.wraps(func)
    def _evaluate_server_model_(server_flex_model, _, *args, **kwargs):
        return func(server_flex_model, *args, **kwargs)

    return _evaluate_server_model_


# def aggregate_weights_tf(func):
#     @functools.wraps(func)
#     def aggregate_weights_tf_(agg_model, weights_key="weights"):
#         aggregated_weights = func(agg_model[weights_key])
#         agg_model["model"].set_weights(aggregated_weights)
#         agg_model[weights_key] = []

#     return aggregate_weights_tf_


# def aggregate_weights_pt(func):
#     @functools.wraps(func)
#     def aggregate_weights_pt_(agg_model):
#         import torch
#         aggregated_weights = func(agg_model["weights"])
#         with torch.no_grad():
#             for old, new in zip(agg_model["model"].parameters(), aggregated_weights):
#                 old.data = torch.from_numpy(new).float()
#         agg_model["weights"] = []

#     return aggregate_weights_pt_


# def train_decorator(func):
#     @functools.wraps(func)
#     def train_model(*args):
#         args = list(args)
#         client_model = args.pop(0)
#         client_data = args.pop(0)
#         if "verbose" in kwargs and kwargs["verbose"] == 1:
#             print("Training model at client.")

#         if "model" in kwargs or "weights" in kwargs:
#             raise ValueError(
#                 "'model' and 'weights' are reserved words in out framework, please, dont't use it."
#             )
#         model = client_model["model"]
#         X_data = client_data.X_data
#         y_data = client_data.y_data
#         func(model, X_data, y_data, *args)

#     return train_model
