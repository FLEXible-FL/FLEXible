import functools


class decorator_base(object):
    def __init__(self, *deco_args, **deco_kwargs):  # Args provided to the decorator
        self.deco_args = deco_args
        self.deco_kwargs = deco_kwargs
        self.aggregated_weights_key = deco_kwargs.get(
            "aggregated_weights_key", "aggregated_weights"
        )
        self.weights_key = deco_kwargs.get("weights_key", "weights")

    def __call__(self, func):
        def _wrap_(self, *args, **kwargs):  # Args when the decorated function is called
            return func(
                *args, **kwargs
            )  # Actual args provided to the function that is decorated

        return _wrap_


class init_server_model(decorator_base):
    def __call__(self, func):
        @functools.wraps(func)
        def _init_server_model_(server_flex_model, _, *args, **kwargs):
            server_flex_model |= func(*args, **kwargs)

        return _init_server_model_


class deploy_server_model(decorator_base):
    def __call__(self, func):
        @functools.wraps(func)
        def _deploy_model_(server_flex_model, clients_flex_models, *args, **kwargs):
            for k in clients_flex_models:
                # Reminder, it is not possible to make assignements here
                clients_flex_models[k] |= func(server_flex_model, *args, **kwargs)

        return _deploy_model_


class collect_clients_weights(decorator_base):
    def __call__(self, func):
        @functools.wraps(func)
        def _collect_weights_(
            aggregator_flex_model, clients_flex_models, *args, **kwargs
        ):
            if self.weights_key not in aggregator_flex_model:
                aggregator_flex_model[self.weights_key] = []
            for k in clients_flex_models:
                client_weights = func(clients_flex_models[k], *args, **kwargs)
                aggregator_flex_model[self.weights_key].append(client_weights)

        return _collect_weights_


class aggregate_weights(decorator_base):
    def __call__(self, func):
        @functools.wraps(func)
        def _aggregate_weights_(aggregator_flex_model, _, *args, **kwargs):
            aggregator_flex_model[self.aggregated_weights_key] = func(
                aggregator_flex_model[self.weights_key], *args, **kwargs
            )
            aggregator_flex_model[self.weights_key] = []

        return _aggregate_weights_


class set_aggregated_weights(decorator_base):
    def __call__(self, func):
        @functools.wraps(func)
        def _deploy_aggregated_weights_(
            aggregator_flex_model, servers_flex_models, *args, **kwargs
        ):
            for k in servers_flex_models:
                func(
                    servers_flex_models[k],
                    aggregator_flex_model[self.aggregated_weights_key],
                    *args,
                    **kwargs
                )

        return _deploy_aggregated_weights_


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
