"""File that contains the primitive functions to build an easy training loop of the federated learning model.

In this file there are some functions that has always the same structure in every Deep Learning Framework,
and some dectorators functions that will help the user to create the training loop.
"""

from copy import deepcopy


def initialize_server_model(flex_model, *args, **kwargs):
    """Function that initialize the model to be trained at the
    server side.

    This function acts as a message handler, that will initialize
    the model at the server side in a client-server architecture.

    To initialize the parameters of the model, you must provide them
    as *args, because the kwargs provided will be kept as clients params
    for the training of the model.

    Example of use using TensorFlow:

        1- First we create the function/class that creates the model:
            def define_model(*args):
                model = "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/2"
                hub_layer = hub.KerasLayer(model, input_shape=[], dtype=tf.string, trainable=True)
                model = tf.keras.Sequential()
                model.add(hub_layer)
                model.add(tf.keras.layers.Dense(16, activation='relu'))
                model.add(tf.keras.layers.Dense(1))
                model.compile(optimizer='adam',
                                loss=tf.losses.BinaryCrossentropy(from_logits=True),
                                metrics=[tf.metrics.BinaryAccuracy(threshold=0.0, name='accuracy')])
                return model

        2- Now we have to create the architecture, in this case
        we will use a client-server architecture. We have to import
        the FlexPool class and the initialize_server_model function.
            from flex.pool import FlexPool
            from flex.pool.primitive_functions import initialize_server_model

        3- Create the client-server architecture:
            flex_pool = FlexPool.client_server_architecture(fed_dataset=flex_dataset, init_func=initialize_server_model, model=define_model, verbose=1, model_params=[])

    In this example we are not using the *args argument as we have fixed params on our model, but we could use them to initialize it.

    Example of use using PyTorch:

        1- First we create the function/class that creates the model:
            class TextClassificationModel(nn.Module):
                def __init__(self, vocab_size, embed_dim, num_class):
                    super(TextClassificationModel, self).__init__()
                    self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
                    self.fc = nn.Linear(embed_dim, num_class)
                    self.init_weights()

                def init_weights(self):
                    initrange = 0.5
                    self.embedding.weight.data.uniform_(-initrange, initrange)
                    self.fc.weight.data.uniform_(-initrange, initrange)
                    self.fc.bias.data.zero_()

                def forward(self, text, offsets):
                    embedded = self.embedding(text, offsets)
                    return self.fc(embedded)

        2- Now we have to create the architecture, as in the TF example,
        we create a client-server architecture:
            from flex.pool import FlexPool
            from flex.pool.primitive_functions import initialize_server_model

            model_params = {
                vocab_size = len(vocab) # Length of the vocab
                embed_dim = 64 # We set the embed dim to 64 for creating a low model, just for the tutorial
                num_class = len(set(train_labels)) # Number of classes on the dataset
            }

            flex_pool = FlexPool.client_server_architecture(fed_dataset=flex_dataset, init_func=initialize_server_model,
                                                model = TextClassificationModel
                                                model_params = model_params
                                                criterion=torch.nn.CrossEntropyLoss(),
                                                optimizer=torch.optim.SGD,
                                                learning_rate=5
                                                )

    Args:
        flex_model (FlexModel): FlexModel where we will allocate
        the model to train.
    """
    if "verbose" in kwargs and kwargs["verbose"] == 1:
        print("Initializing server model.")
        del kwargs["verbose"]

    model = kwargs["model"](kwargs["model_params"])
    flex_model["model"] = model
    del kwargs["model"]
    del kwargs["model_params"]
    if len(kwargs):
        for k_arg, v_arg in kwargs.items():
            flex_model[k_arg] = v_arg


def deploy_model_to_clients(server_model, clients_model, *args, **kwargs):
    """Function that deploy the Model at the server to each FlexModel
    in the clients.

    This functions acts as a message handler, that will deploy the
    model to train to the clients, and must be used from the server
    to the clients. Supposing we have a FlexPool with one server and
    another FlexPool of clients, we will use this function as follows
    in a client-server architecture:

    from flex.pool.primitive functions import deploy_model_to_clients

    flex_pool = FlexPool.client_server_architecture(fed_dataset=flex_dataset, init_func=initialize_server_model)

    clients = flex_pool.clients #Select the pool of clients
    server = flex_pool.servers #Select the pool of servers

    server.map(deploy_model_to_clients, clients)

    If we didnt take the pools separately as shown in the above example,
    we can do the same with the full pool. First we load pool with a
    client-server architecture:

    from flex.pool.primitive functions import deploy_model_to_clients

    flex_pool = FlexPool.client_server_architecture(fed_dataset=flex_dataset, init_func=initialize_server_model)

    flex_pool.server.map(deploy_model_to_clients, flex_pool.clients)

    In this last case, we didn't separate the pools, because we don't
    need to do it, but we highly recommend it for a better understanding
    of the code that is being developed.

    Notes: To see how to create a client-server architecture go to the client_server_architecture
    function at FlexPool, or check the tutorials on how to federate
    each TensorFlow/Pytorch models.

    Args:
        server_model (Union[TensorFlow.Model, PyTorch.Model]): The TensorFlow/PyTorch/OtherFramework
        model that will be copied to the clients.
        clients_model (List[FlexModel]): A list of FlexModel that represent each client.
    """
    if "verbose" in kwargs and kwargs["verbose"] == 1:
        print("Initializing model at a client")
        del kwargs["verbose"]

    for client_id in clients_model:
        clients_model[client_id] = deepcopy(server_model)


def train(client_model, client_data, *args, **kwargs):
    # TODO: Fulfill this function to work better for PyTorch.
    # TODO: Improve documentation.
    """Function to train train a model at client level. This
    function will use the model allocated at the client level,
    and then will feed the client's data to the model in orther
    to train it.

    We use the args and kwargs arguments to give it to the train
    function of the model. Each model will have it's own train
    function, i.e., a TensorFlow model will call .fit() method,
    so we will call the method as follows:
    model.fit(X_data, y_data, *args, **kwargs)

    Args:
        client_model (_type_): _description_
        client_data (_type_): _description_
    """
    if "verbose" in kwargs and kwargs["verbose"] == 1:
        print("Training model at client.")

    if "model" in kwargs or "weights" in kwargs:
        raise ValueError(
            "'model' and 'weights' are reserved words in out framework, please, dont't use it."
        )

    model = client_model["model"]
    X_data = client_data.X_data
    y_data = client_data.y_data
    if "func" in kwargs:
        # PyTorch models need it's own function
        func = kwargs["func"]
        del kwargs["func"]
        for item in client_model:
            if item not in ["model", "weights"]:
                kwargs.append(item)
        func(model, X_data, y_data, *args, **kwargs)

        def wrapper(func, model, client_data, *args, **kwargs):
            func(model, client_data, args, kwargs)

        return wrapper
    else:
        model.fit(X_data, y_data, *args, **kwargs)


def collect_weights(client_model, aggregator_model, **kwargs):
    """Function to collects the weights for the client's model
    and 'send' it to the aggregator.

    Args:
        client_model (FlexModel): Client's FlexModel
        aggregator_model (FlexModel): Aggregator's FlexModel

    This function will collect the weights from the client's model,
    simulating that the clients are sending their model weights to
    the aggregator.

    You can provide a custom func to aggregate the weights of your
    model if you want to collect just part of them, if not,
    it's not necessary, as we offer functions to get the weights
    from the models.

    Example of use assuming you are using a client-server architechture:

        from flex.pool.primitive_functions import collect_weights

        clients = flex_pool.clients
        aggregator = flex_pool.aggregators

        clients.map(collect_weights, aggregator)

    Example of use using the FlexPool without separating clients
    and aggregator, and following a client-server architechture.

        from flex.pool.primitive_functions import collect_weights

        flex_pool.clients.map(collect_weights, flex_pool.aggregators)

    Note: Right now we offer support to TensorFlow and PyTorch
    models, as they are the most used frameworks, but you can
    collect the weights of your own model providing a func
    as parameter in the kwargs.
    """
    if "verbose" in kwargs and kwargs["verbose"] == 1:
        print("Collecting weights.")

    if "weights" not in aggregator_model["server"]:
        aggregator_model["server"]["weights"] = []

    if "func" in kwargs:
        func = kwargs["func"]
        client_weights = func(client_model["model"])
    else:
        import tensorflow as tf
        import torch.nn as nn

        if isinstance(client_model["model"], tf.Module):

            def tensorflow_weights_collector(client_model):
                return client_model.get_weights()

            client_weights = tensorflow_weights_collector(client_model["model"])
        elif isinstance(client_model["model"], nn.Module):

            def pytorch_weights_collector(client_model):
                return [param.cpu().data.numpy() for param in client_model.parameters()]

            client_weights = pytorch_weights_collector(client_model["model"])
        else:
            print(
                "Warning, using your custom model, we can't provide support to this section, so be sure that your function only needs the client_model['model'].get_model() as parameter."
            )
            client_weights = func(client_model["model"].get_model())
    aggregator_model["server"]["weights"].append(client_weights)


def aggregate_weights(agg_model, *args, **kwargs):
    """Function to aggregate the collected weights using a
    personalized aggregation function.

    Args:
        agg_model (FlexModel): The FlexModel for the aggregator
        func_aggregate (Callabe): A function to aggregate the weights.
        func (Callabe, Optional): A function to set the weights
        in a personalized way.

    Raises:
        ValueError: Error when not providing the func_aggregate function, i.e., the aggregation method.
        ValueError: Error if using a custom model and not providing the function to aggregate the weights. Should be provided as func param.

    This function will aggregate all the collected weights
    at the aggregator. This function will be called by the aggregator.

    You have to provide the aggregation function that will aggreate
    the weights. You can create your own personalized function to do it
    with both TensorFlow and Pytorch models.

    Also you can provide a function to set only some weights to the model and not all the weights.

    Example of use assuming you are using a client-server architechture:

        from flex.pool.primitive_functions import aggregate_weights

        aggregator = flex_pool.aggregators

        aggregator.map(aggregate_weights)

    Example of use using the FlexPool without separating clients
    and aggregator, and following a client-server architechture.

        from flex.pool.primitive_functions import aggregate_weights

        flex_pool.aggregators.map(aggregate_weights)

    Note: Right now we offer support to TensorFlow and PyTorch
    models, as they are the most used frameworks, but you can
    aggregate the weights of your own model providing a func
    as parameter in the kwargs.
    """
    if "verbose" in kwargs and kwargs["verbose"] == 1:
        print("Aggregating weights")

    if "func_aggregate" not in kwargs:
        raise ValueError(
            "func_aggregate parameter is required, please provide it to aggregate the weights. The function will be called as follows: func_aggregate(weights)."
        )

    aggregated_weights = kwargs["func_aggregate"](agg_model["weights"])

    if "func" in kwargs:
        func = kwargs["func"]
        func(agg_model["model"], aggregated_weights)
    else:
        import tensorflow as tf
        import torch

        if isinstance(agg_model["model"], tf.Module):
            agg_model["model"].set_weights(aggregated_weights)
        elif isinstance(agg_model["model"], torch.nn.Module):
            with torch.no_grad():
                for old, new in zip(
                    agg_model["model"].parameters(), aggregated_weights
                ):
                    old.data = torch.from_numpy(new).float()
        else:
            print(
                "Warning, using your custom model, we can't provide support to this section, so be sure that your function only needs the client_model['model'].get_model() as parameter."
            )

            if "func" not in kwargs:
                raise ValueError(
                    'When using your custom model, please provide a function to set the model params. Provide as param "func".'
                )

            func = kwargs["func"]
            func(agg_model["model"])
    agg_model["weights"] = []


def deploy_global_model_to_clients(server_model, clients_models, *args, **kwargs):
    """Function that deploy the global model to the clients.

    Args:
        server_model (FlexModel): The FlexModel from the server
        clients_models (List[FlexModel]): List with the FlexModels from the clients.

    Raise:
        ValueError: If using a custom model and not providing the 'func' argument to
        deploy the global model.

    This function will deploy the server model to the clients once the
    weights have been aggregated.

    If using a custom model, you will have to provide the function that will
    deploy the model to the clients.

    Example of use assuming you are using a client-server architechture:

        from flex.pool.primitive_functions import deploy_global_model_to_clients

        clients = flex_pool.clients
        server = flex_pool.servers

        server.map(deploy_global_model_to_clients, clients)

    Example of use using the FlexPool without separating clients
    and aggregator, and following a client-server architechture.

        from flex.pool.primitive_functions import deploy_global_model_to_clients

        flex_pool.servers.map(deploy_global_model_to_clients, flex_pool.clients)

    Note: Right now we offer support to TensorFlow and PyTorch
    models, as they are the most used frameworks, but you can
    deploy the global mdel of your own model providing a func
    as parameter in the kwargs.
    """
    if "verbose" in kwargs and kwargs["verbose"] == 1:
        print("Deploying the global model on the clients.")

    import tensorflow as tf
    import torch

    if isinstance(server_model["model"], tf.Module):
        aggregated_weights = server_model["model"].get_weights()
        for client_model in clients_models:
            clients_models[client_model]["model"].set_weights(aggregated_weights)
    elif isinstance(server_model["model"], torch.nn.Module):
        aggregated_weights = [
            param.cpu().data.numpy() for param in server_model["model"].parameters()
        ]
        with torch.no_grad():
            for client_model in clients_models:
                for old, new in zip(
                    clients_models[client_model]["model"].parameters(),
                    aggregated_weights,
                ):
                    old.data = torch.from_numpy(new).float()
    else:
        print(
            "Warning, using your custom model, we can't provide support to this section, so be sure that your function only needs the client_model['model'].get_model() as parameter."
        )
        if "func" not in kwargs:
            raise ValueError(
                'When using your custom model, pleae provide a function to deploy the global model. Provide as param "func'
            )
        func = kwargs["func"]
        func(server_model["model"], clients_models)


def evaluate_model(model, data, *args, **kwargs):
    """Function that evaluate the global model at client or server level.

    Args:
        model (FlexModel): FlexModel with the model to
        data (FlexDataObject): Client's data to test the model with. Might be None at server node.

    This functions evaluate the global model on the clients data and on
    the global test data.

    Example of use assuming you are using a client-server architechture
    and a TensorFlow model:

        from flex.pool.primitive_functions import evaluate_model

        clients = flex_pool.clients
        server = flex_pool.servers

        server.map(evaluate_model, test_examples=test_examples, test_labels=test_labels)

        clients.map(evaluate_model, test_examples=test_examples, test_labels=test_labels)

    Example of use using the FlexPool without separating clients
    and aggregator, and following a client-server architechture
    and a TensorFlow model:

        from flex.pool.primitive_functions import evaluate_model

        flex_pool.servers.map(evaluate_model, test_examples=test_examples, test_labels=test_labels)

        flex_poolclients.map(evaluate_model, test_examples=test_examples, test_labels=test_labels)

    """
    X_test, y_test = kwargs["test_examples"], kwargs["test_labels"]
    del kwargs["test_examples"]
    del kwargs["test_labels"]

    if "eval_func" not in kwargs:
        import tensorflow as tf

        assert isinstance(model["model"], tf.Module)

        def eval_func(model, X_test, y_test, **kwargs):
            """Basic evaluate function for a TensorFlow Model"""
            return model["model"].evaluate(X_test, y_test)

    else:
        eval_func = kwargs["eval_func"]
        del kwargs["eval_func"]
    if data is not None:
        print("Evaluating model at client.")
        results_local = eval_func(model, data.X_data, data.y_data, **kwargs)
        print(f"Results at client on client's data: {results_local}")
    else:
        print("Evaluating model at server")

    results = eval_func(model, X_test, y_test, **kwargs)
    print(f"Results on test data: {results}")
