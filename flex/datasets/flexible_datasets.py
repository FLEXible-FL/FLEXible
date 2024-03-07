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
