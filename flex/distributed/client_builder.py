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

from typing import Callable, List

import tensorly as tl
from numpy import ndarray

from flex.data.dataset import Dataset
from flex.distributed.client import Client
from flex.model import FlexModel
from flex.pool.aggregators import set_tensorly_backend


class FlexibleClient(Client):
    def __init__(
        self,
        flex_model: FlexModel,
        dataset: Dataset,
        train: Callable,
        collect: Callable,
        set_weights: Callable,
        eval: Callable,
        eval_dataset: Dataset,
    ) -> None:
        self._train = train
        self._collect = collect
        self._setweights = set_weights
        self._eval = eval
        super().__init__(dataset=dataset, model=flex_model, eval_dataset=eval_dataset)

    def train(self, model: FlexModel, data: Dataset):
        return self._train(model, data)

    def eval(self, model: FlexModel, data: Dataset):
        return self._eval(model, data)

    def get_weights(self, model: FlexModel) -> List[ndarray]:
        return self._collect(model)

    def set_weights(self, model: FlexModel, weights: List[ndarray]):
        return self._setweights(model, weights)


class ClientBuilder:
    """
    A builder class for creating a client that allows reusing the wrapped FLEX functions.

    This class provides a convenient way to build a client object by setting the necessary components
    such as the flex model, dataset, training function, evaluation function, and other required functions.

    Example usage:
    ```
    client = ClientBuilder()
        .model(flex_model)
        .dataset(dataset)
        .train(train_function)
        .collect_weights(collect_weights_function)
        .set_weights(set_weights_function)
        .eval(eval_function, eval_dataset)
        .build()
    ```

    Once all the required components are set, the `build()` method can be called to create a `Client` object.

    Note: It is important to set all the required components before calling the `build()` method.

    """

    def __init__(self) -> None:
        self._flex_model = None
        self._dataset = None
        self._train = None
        self._collect = None
        self._setweights = None
        self._eval = None
        self._eval_dataset = None

    def train(self, train: Callable) -> "ClientBuilder":
        """
        Set the training function for the client.

        Args:
        ----
            train (Callable): The training function.

        Returns:
        -------
            ClientBuilder: The updated client builder object.

        """
        self._train = train
        return self

    def eval(self, eval: Callable, eval_dataset: Dataset) -> "ClientBuilder":
        """
        Set the evaluation function and evaluation dataset for the client.

        Args:
        ----
            eval (Callable): The evaluation function.
            eval_dataset (Dataset): The evaluation dataset.

        Returns:
        -------
            ClientBuilder: The updated client builder object.

        """
        self._eval = eval
        self._eval_dataset = eval_dataset
        return self

    def collect_weights(self, collect_weights: Callable) -> "ClientBuilder":
        """
        Set the collect weights function for the client.

        Args:
        ----
            collect_weights (Callable): The collect weights function.

        Returns:
        -------
            ClientBuilder: The updated client builder object.

        Raises:
        ------
            ValueError: If the function passed as an argument is not wrapped by `collect_clients_weights`.

        """
        try:
            self._collect = collect_weights.__wrapped__

            def _collect_to_numpy_(model: FlexModel):
                weights = collect_weights.__wrapped__(model)
                set_tensorly_backend(weights)
                return [tl.to_numpy(tl.tensor(t)) for t in weights]

            self._collect = _collect_to_numpy_
            return self

        except AttributeError:
            raise ValueError(
                "Function passed as argument must be wrapped by collect_clients_weights"
            ) from None

    def set_weights(self, set_weights: Callable) -> "ClientBuilder":
        """
        Set the set weights function for the client. The recieved weights are numpy `ndarray`.

        Args:
        ----
            set_weights (Callable): The set weights function.

        Returns:
        -------
            ClientBuilder: The updated client builder object.

        Raises:
        ------
            ValueError: If the function passed as an argument is not wrapped by `set_aggregated_weights`.

        """
        try:
            self._setweights = set_weights.__wrapped__
            return self
        except AttributeError:
            raise ValueError(
                "Function passed as argument must be wrapped by set_aggregated_weights"
            ) from None

    def dataset(self, dataset: Dataset) -> "ClientBuilder":
        """
        Set the dataset for the client.

        Args:
        ----
            dataset (Dataset): The dataset.

        Returns:
        -------
            ClientBuilder: The updated client builder object.

        """
        self._dataset = dataset
        return self

    def model(self, flex_model: FlexModel) -> "ClientBuilder":
        """
        Set the flex model for the client.

        Args:
        ----
            flex_model (FlexModel): The flex model.

        Returns:
        -------
            ClientBuilder: The updated client builder object.

        """
        self._flex_model = flex_model
        return self

    def build_model(self, build_server_model: Callable) -> "ClientBuilder":
        """
        Builds the client model by calling the provided `build_server_model` function.

        Args:
        ----
            build_server_model (Callable): A function that builds the server model.

        Returns:
        -------
            ClientBuilder: The instance of the ClientBuilder class.

        """
        return self.model(build_server_model.__wrapped__())

    def build(self) -> Client:
        """
        Build and return the client object.

        Returns
        -------
            Client: The client object.

        Raises
        ------
            AssertionError: If any of the required components are not set.

        """
        assert self._flex_model is not None, "model must be set"
        assert self._dataset is not None, "dataset must be set"
        assert self._train is not None, "train function must be set"
        assert self._collect is not None, "collect function must be set"
        assert self._setweights is not None, "set_weights function must be set"
        assert self._eval is not None, "eval function must be set"
        assert self._eval_dataset is not None, "eval dataset must be set"
        return FlexibleClient(
            self._flex_model,
            self._dataset,
            self._train,
            self._collect,
            self._setweights,
            self._eval,
            self._eval_dataset,
        )
