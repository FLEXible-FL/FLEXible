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


from inspect import getmembers, isfunction

from flex.datasets import federated_datasets, standard_datasets


def list_datasets():
    """
    Function that returns the available datasets in FLEXible.
    Returns: A list with the available datasets.
    -------

    """
    return list(ds_invocation.keys())


def load(name, *args, **kwargs):
    """
    Function that loads every dataset builtin into the framework
    Parameters
    ----------
    name


    Returns
    -------

    """
    try:
        caller = ds_invocation[name.lower()]
    except KeyError as e:
        raise ValueError(
            f"{name} is not available. Please, check available datasets using list_datasets "
            f"function. "
        ) from e

    return caller(*args, **kwargs)


ds_invocation = dict(
    getmembers(standard_datasets, isfunction)
    + getmembers(federated_datasets, isfunction)
)
