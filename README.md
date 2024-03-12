![](https://twemoji.maxcdn.com/v/latest/72x72/1f938.png)

# FLEXible

[![Tests](https://github.com/FLEXible-FL/FLEX-framework/actions/workflows/pytest.yml/badge.svg)](https://github.com/FLEXible-FL/FLEX-framework/actions/workflows/pytest.yml)
[![Linter](https://github.com/FLEXible-FL/FLEX-framework/actions/workflows/trunk.yml/badge.svg)](https://github.com/FLEXible-FL/FLEX-framework/actions/workflows/trunk.yml)

FLEXible (Federated Learning Experiments) is an open source Federated Learning (FL) framework that provides a set of tools and utilities to work with deep learning and machine learning models in a federated scenario. It is designed to federate data and models in a simple and easy way, and it is compatible with the most popular deep learning frameworks such as PyTorch and TensorFlow. It also provides a set of federated datasets to test the models.

FLEXible let the user to customize the federated scenario, from the bottom to the top. FLEXible has the following tools to carry out the federated learning experiments:

- Datasets: FLEXible provides a set of federated datasets to test the models. Some datasets are: MNIST, CIFAR10, Shakespeare, etc.
- Data: FLEXible provides a set of tools to federate your own data. You can import your own data our even import the data from other libraries such as `torchvision`, `torchtext`, `datasets` or `tensorflow_datasets`.
- Architecture: In FLEXible you can create custom federated architectures. You can quickly deploy a client-server architecture or a peer-to-peer architecture, or easily create your own federated architecture.
- Roles: FLEXible provides a set of roles to define the federated scenario. Usually, you will work with the `Server`, `Aggregator` and the `Client` roles, but you can create nodes with multiple roles, such as `Server` and `Client` at the same time, or `Server` and `Aggregator` at the same time. The last one is used in the client-server architecture.
- FLEXible defines the [`FlexPool`](https://github.com/FLEXible-FL/FLEXible/blob/main/flex/pool/pool.py) as the orchestrator of the federated scenario.
- FLEXible provides its own [decorators](https://github.com/FLEXible-FL/FLEXible/blob/main/flex/pool/decorators.py) to define the federated functions. Also, FLEXible provides a set of primitives for different frameworks.[PyTorch primitives](https://github.com/FLEXible-FL/FLEXible/blob/main/flex/pool/primitives_pt.py) and [TensorFlow primitives](https://github.com/FLEXible-FL/FLEXible/blob/main/flex/pool/primitives_tf.py), that let the user adapt their centralized experiments to a federated scenario.
- FLEXible algo provides some [aggregators](https://github.com/FLEXible-FL/FLEXible/blob/main/flex/pool/aggregators.py), such as FedAVG or WeightedFedAVG, but you can create your own aggregators.

## Installation

We recommend Anaconda/Miniconda as the package manager. To install the package, you can use the following commands:

### Using pip

```bash
pip install flexible-fl
```

### Download the repository and install it locally

First download the repository:

```bash
git clone git@github.com:FLEXible-FL/FLEXible.git
cd FLEXible
```

Then, install the package:

- Without support for any particular framework

    ```bash
    pip install -e .
    ```

- With only pytorch support:

    ```bash
    pip install -e ".[pt]"
    ```

- With only tensorflow support:

    ```bash
    pip install -e ".[tf]"
    ```

- In order to install this repo locally for development:

    ```bash
    pip install -e ".[develop]"
    ```

## Getting started

To get started with **FLEXible**, you can check the [notebooks](https://github.com/FLEXible-FL/FLEXible/tree/main/notebooks) available in the repository. These notebooks have examples for how to federate data, or how to integrate deep learning frameworks such as PyTorch or TensorFlow. Some notebooks are:
- [Create custom federated architectures using FLEXible](https://github.com/FLEXible-FL/FLEXible/blob/main/notebooks/Create%20custom%20architectures%20using%20FLEXible.ipynb).
- [Federate data with FLEXible](https://github.com/FLEXible-FL/FLEXible/blob/main/notebooks/Federating%20data%20with%20FLEXible.ipynb).
- [Federate a simple neural network with FLEXible using the FLEXible primitives and using the MNIST dataset](https://github.com/FLEXible-FL/FLEXible/blob/main/notebooks/Federated%20MNIST%20PT%20example%20with%20simplified%20primitives.ipynb).
- [Federate a simple neural network with FLEXible using the FLEXible primitives and using the IMDb dataset](https://github.com/FLEXible-FL/FLEXible/blob/main/notebooks/Federated%20Text%20Classification%20TF%20with%20simplified%20primitives.ipynb).
- [Vertical federated learning using FLEXible with a tabular dataset](https://github.com/FLEXible-FL/FLEXible/blob/main/notebooks/Vertical%20Federated%20Adult-income%20Classification%20PT.ipynb).

## Documentation

To build the documentation you will need to `sphinx` and `sphinx-rtd-theme`. You can install them using pip:

```bash
pip install sphinx sphinx-rtd-theme
```

Once you have installed the dependencies, you can build the documentation using the following commands:

```bash
cd docs/
make clean html # Generate the documentation in HTML format.
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package, please cite the following paper:

``` TODO: Add citation ```

### Others
**Authors order**: The order of the authors was chosen by asking ChatGPT to make a program to do it.

The conversation was:

Q: "Write a program in Python that chooses between two people at random, Alberto and Daniel, the order between them given a random seed that you choose. Also, run the program and tell me the result."

A: "Here is a Python program that randomly selects the order between two people, Alberto and Daniel, using a predefined random seed.

```python
import random

# Predefined random seed
random.seed(123)

# List with people's names
people = ["Alberto", "Daniel"]

# Random choice of order
order = random.sample(people, k=2)

# Print the result
print("The selected order is:", order)
```

When running this program, the result may vary depending on the random seed selected. However, since we have set a specific seed (`random.seed(123)`), the result will be the same every time it is run with that seed.

Example result:

```bash
The selected order is: ['Daniel', 'Alberto']
```

Remember that if you change the random seed or run the program at another time, the result will be different."
