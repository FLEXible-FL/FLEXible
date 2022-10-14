import functools


def init_server_model_decorator(func):
    @functools.wraps(func)
    def initialize_server_model(*args, **kwargs):
        flex_model = args[0]
        if "verbose" in kwargs and kwargs["verbose"] == 1:
            print("Initializing server model.")
            del kwargs["verbose"]
        model = func(kwargs["model_params"])
        flex_model["model"] = model
        del kwargs["model_params"]
        if len(kwargs):
            for k_arg, v_arg in kwargs.items():
                flex_model[k_arg] = v_arg

    return initialize_server_model


def collector_decorator(func):
    @functools.wraps(func)
    def collector_weights(*args, **kwargs):
        client_model = args[0]
        aggregator_model = args[1]
        if "verbose" in kwargs and kwargs["verbose"] == 1:
            print("Collecting weights.")

        if "weights" not in aggregator_model["server"]:
            aggregator_model["server"]["weights"] = []

        client_weights = func(client_model["model"])
        aggregator_model["server"]["weights"].append(client_weights)

    return collector_weights


def aggregator_decorator_tf(func):
    @functools.wraps(func)
    def aggregate_weights(*args, **kwargs):
        agg_model = args[0]
        if "verbose" in kwargs and kwargs["verbose"] == 1:
            print("Aggregating weights")
        aggregated_weights = func(agg_model["weights"])
        agg_model["model"].set_weights(aggregated_weights)
        agg_model["weights"] = []

    return aggregate_weights


def aggregator_decorator_pt(func):
    @functools.wraps(func)
    def aggregate_weights(*args, **kwargs):
        agg_model = args[0]
        if "verbose" in kwargs and kwargs["verbose"] == 1:
            print("Aggregating weights")

        aggregated_weights = func(agg_model["weights"])
        import torch

        with torch.no_grad():
            for old, new in zip(agg_model["model"].parameters(), aggregated_weights):
                old.data = torch.from_numpy(new).float()
        agg_model["weights"] = []

    return aggregate_weights


def train_decorator(func):
    @functools.wraps(func)
    def train_model(*args, **kwargs):
        args = list(args)
        client_model = args.pop(0)
        client_data = args.pop(0)
        if "verbose" in kwargs and kwargs["verbose"] == 1:
            print("Training model at client.")

        if "model" in kwargs or "weights" in kwargs:
            raise ValueError(
                "'model' and 'weights' are reserved words in out framework, please, dont't use it."
            )
        model = client_model["model"]
        X_data = client_data.X_data
        y_data = client_data.y_data
        func(model, X_data, y_data, *args, **kwargs)

    return train_model


def deploy_decorator(func):
    @functools.wraps(func)
    def deploy_global_model_to_clients(*args, **kwargs):
        if "verbose" in kwargs and kwargs["verbose"] == 1:
            print("Deploying the global model on the clients.")
        args = list(args)
        print(f"Args: {args}")
        server_model = args.pop(0)
        clients_models = args.pop(0)
        func(server_model["model"], clients_models)

    return deploy_global_model_to_clients
