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
import contextlib
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
from cardinality import count

from flex.data.lazy_indexable import LazyIndexable


@dataclass(frozen=True)
class Dataset:
    """Class used to represent the dataset from a node in a Federated Learning enviroment.

    Attributes
    ----------
    X_data: LazyIndexable
        A numpy.array containing the data for the node.
    y_data: LazyIndexable
        A numpy.array containing the labels for the training data. Can be None if working
        on an unsupervised learning task. Default None.
    """

    X_data: LazyIndexable = field(init=True)
    y_data: Optional[LazyIndexable] = field(default=None, init=True)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer)):
            return (
                self.X_data[index],
                self.y_data[index] if self.y_data is not None else None,
            )
        elif isinstance(index, (slice, list)):
            return Dataset(
                self.X_data[index],
                self.y_data[index] if self.y_data is not None else None,
            )
        else:
            raise IndexError(
                f"Indexing with element {index} of type {type(index)} is not supported"
            )

    def __iter__(self):
        return zip(
            self.X_data,
            self.y_data if self.y_data is not None else [None] * len(self),
        )

    def to_torchvision_dataset(self, **kwargs):
        """This function transforms a Dataset into a Torchvision dataset object

        Returns:
        --------
            torvhcision.datasets.VisionDataset: a torchvision dataset with the contents of datasets. \
                Note that transforms should be pased as arguments.
        """
        from .dataset_pt_utils import DefaultVision

        return DefaultVision(self, **kwargs)

    def to_tf_dataset(self):
        """This function is an utility to transform a Dataset object to a tensorflow.data.Dataset object

        Returns:
        --------
            tensorflow.data.Dataset: tf dataset object instanciated using the contents of a Dataset
        """
        from tensorflow import type_spec_from_value
        from tensorflow.data import Dataset as tf_Dataset

        return tf_Dataset.from_generator(
            self.__iter__,
            output_signature=(
                type_spec_from_value(self[0][0]),
                type_spec_from_value(self[0][1]),
            ),
        )

    def to_numpy(self, x_dtype=None, y_dtype=None):
        """Function to return the FlexDataObject as numpy arrays."""

        if self.y_data is None:
            return self.X_data.to_numpy(dtype=x_dtype)
        else:
            return self.X_data.to_numpy(x_dtype), self.y_data.to_numpy(dtype=y_dtype)

    def to_list(self):
        """Function to return the FlexDataObject as list."""
        if self.y_data is None:
            return self.X_data.tolist()
        else:
            return self.X_data.tolist(), self.y_data.tolist()

    @classmethod
    def from_torchvision_dataset(cls, pytorch_dataset):
        """Function to convert an object from torchvision.datasets.* to a FlexDataObject.

        Args:
        -----
            pytorch_dataset (torchvision.datasets.*): a torchvision dataset.

        Returns:
        --------
            Dataset: a FlexDataObject which encapsulates the dataset.
        """

        from flex.data.pluggable_datasets import PluggableTorchvision

        if pytorch_dataset.__class__.__name__ not in PluggableTorchvision:
            warnings.warn(
                "The input dataset and arguments are not explicitly supported, therefore they might not work as expected.",
                RuntimeWarning,
            )

        length = count(pytorch_dataset)

        X_data = LazyIndexable((x for x, _ in pytorch_dataset), length=length)
        y_data = LazyIndexable((y for _, y in pytorch_dataset), length=length)

        return cls(X_data=X_data, y_data=y_data)

    @classmethod
    def from_tfds_image_dataset(cls, tfds_dataset):
        """Function to convert a dataset from tensorflow_datasets to a FlexDataObject.

        Args:
        -----
            tdfs_dataset (tf.data.Datasets): a tf dataset

        Returns:
        --------
            Dataset: a FlexDataObject which encapsulates the dataset.
        """

        if not isinstance(tfds_dataset, tuple):
            # unbatch if required
            with contextlib.suppress(ValueError):
                tfds_dataset = tfds_dataset.unbatch()

            # After unbatching, we can't get the length, so we have to get it.
            # To get the length, we use count.
            length = count(tfds_dataset.as_numpy_iterator())
            X_data = LazyIndexable(
                (x for x, _ in tfds_dataset.as_numpy_iterator()), length=length
            )
            y_data = LazyIndexable(
                (y for _, y in tfds_dataset.as_numpy_iterator()), length=length
            )
        else:
            X_data = LazyIndexable(iter(tfds_dataset[0]), length=len(tfds_dataset[0]))
            y_data = LazyIndexable(iter(tfds_dataset[1]), length=len(tfds_dataset[1]))

        return cls(X_data=X_data, y_data=y_data)

    @classmethod
    def from_tfds_text_dataset(
        cls, tfds_dataset, X_columns: list = None, label_columns: list = None
    ):
        """Function to convert a dataset from tensorflow_datasets to a FlexDataObject.

        Args:
        -----
            tdfs_dataset (tf.data.Datasets): a tf dataset loaded.
            X_columns (list): List containing the features (input) of the model.
            label_columns (list): List containing the targets of the model.

        Returns:
        --------
            Dataset: a FlexDataObject which encapsulates the dataset.
        """
        from tensorflow.python.data.ops.dataset_ops import PrefetchDataset

        if isinstance(tfds_dataset, PrefetchDataset):
            # First case: Users used load func with batch_size != -1 or without indicating the batch_size
            length = len(tfds_dataset)
            if not isinstance(tfds_dataset, tuple):
                with contextlib.suppress(ValueError):
                    tfds_dataset.unbatch()
            if X_columns is None:
                X_data_generator = iter(tfds_dataset.as_numpy_iterator())
            elif len(X_columns) == 1:
                X_data_generator = (
                    tuple(map(row.get, X_columns))[0]
                    for row in tfds_dataset.as_numpy_iterator()
                )
            else:
                X_data_generator = (
                    tuple(map(row.get, X_columns))
                    for row in tfds_dataset.as_numpy_iterator()
                )
            X_data = LazyIndexable(X_data_generator, length=length)

            if label_columns is None:
                y_data = None
            elif len(label_columns) == 1:
                y_data_generator = (
                    tuple(map(row.get, label_columns))[0]
                    for row in tfds_dataset.as_numpy_iterator()
                )
                y_data = LazyIndexable(y_data_generator, length=length)
            else:
                y_data_generator = (
                    tuple(map(row.get, label_columns))
                    for row in tfds_dataset.as_numpy_iterator()
                )
                y_data = LazyIndexable(y_data_generator, length=length)
        else:  # User used batch_size=-1 when using the load function
            if X_columns is None:
                X_data_generator = iter(map(tfds_dataset.get, tfds_dataset.keys()))
            else:
                X_data_generator = iter(map(tfds_dataset.get, X_columns))
            X_data = LazyIndexable(X_data_generator, length=len(tfds_dataset))

            if label_columns is None:
                y_data = None
            else:
                y_data_generator = iter(map(tfds_dataset.get, label_columns))
                y_data = LazyIndexable(y_data_generator, length=len(tfds_dataset))

        return cls(X_data=X_data, y_data=y_data)

    @classmethod
    def from_huggingface_dataset(
        cls,
        hf_dataset,
        X_columns: list = None,
        label_columns: list = None,
    ):
        """Function to conver an arrow dataset from the Datasets package (HuggingFace datasets library)
        to a FlexDataObject.

        Args:
        -----
            hf_dataset (Union[datasets.arrow_dataset.Dataset, str]): a dataset from the dataset library.
            If a string is recieved, it will load the dataset from the HuggingFace repository. When a
            string is given, the split has to be specified in the str variable as follows:
            'dataset;split'. Also, if the string contains a subset, for those datasets that have
            multiple subsets for differents tasks, it may be given as follow: 'dataset;subset;split',
            so we can download the dataset and the desired subset and split.
            X_columns (list): List containing the features names for training the model
            label_columns (list): List containing the name or names of the label column

        Returns:
        --------
            Dataset: a FlexDataObject which encapsulates the dataset.
        """
        from flex.data.pluggable_datasets import PluggableHuggingFace

        try:
            name_checker = ""
            if isinstance(hf_dataset, str):
                from datasets import load_dataset

                hf_dataset = hf_dataset.split(";")
                if len(hf_dataset) == 2:
                    name, split = hf_dataset
                    subset = None
                elif len(hf_dataset) == 3:
                    name, subset, split = hf_dataset
                try:
                    hf_dataset = (
                        load_dataset(name, split=split)
                        if subset is None
                        else load_dataset(
                            name, subset, split=split, ignore_verifications=True
                        )
                    )
                except Exception as err:
                    print(
                        f"Couldn't download the dataset from the HuggingFace datasets: {err}"
                    )

                name_checker = (
                    f"{name.upper()}_{subset.upper()}_HF"
                    if subset is not None
                    else f"{name.upper()}_HF"
                )
            else:
                name_checker = hf_dataset.info.builder_name
            if name_checker not in PluggableHuggingFace.__members__.keys():
                warnings.warn(
                    "The input dataset and arguments are not explicitly supported, therefore they might not work as expected.",
                    RuntimeWarning,
                )

        except Exception:
            warnings.warn(
                "The input dataset doesn't have the property dataset.info.builder_name or the str format is not correct, so we can't check if is supported or not. Therefore, it might not work as expected.",
                RuntimeWarning,
            )
        length = count(hf_dataset)

        if X_columns is None:
            X_data_generator = iter(
                zip(*map(hf_dataset.__getitem__, hf_dataset.features))
            )
        elif len(X_columns) == 1:
            X_data_generator = (
                i for x in map(hf_dataset.__getitem__, X_columns) for i in x
            )
        else:
            X_data_generator = iter(zip(*map(hf_dataset.__getitem__, X_columns)))

        X_data = LazyIndexable(X_data_generator, length=length)

        if label_columns is None:
            y_data = None
        elif len(label_columns) == 1:
            y_data_generator = (
                i for x in map(hf_dataset.__getitem__, label_columns) for i in x
            )
            y_data = LazyIndexable(y_data_generator, length=length)
        else:
            y_data_generator = iter(zip(*map(hf_dataset.__getitem__, label_columns)))
            y_data = LazyIndexable(y_data_generator, length=length)
        return cls(X_data=X_data, y_data=y_data)

    @classmethod
    def from_torchtext_dataset(cls, pytorch_text_dataset):
        """Function to convert an object from torchtext.datasets.* to a FlexDataObject.
            It is mandatory that the dataset contains at least the following transform:
            torchtext.transforms.ToTensor()

        Args:
        -----
            pytorch_text_dataset (torchtext.datasets.*): a torchtext dataset

        Returns:
        --------
            Dataset: a FlexDataObject which encapsulates the dataset.
        """

        from flex.data.pluggable_datasets import PluggableTorchtext

        if pytorch_text_dataset.__class__.__name__ not in PluggableTorchtext:
            warnings.warn(
                "The input dataset and arguments are not explicitly supported, therefore they might not work as expected.",
                RuntimeWarning,
            )
        try:
            length = len(pytorch_text_dataset)
        except TypeError:
            y_data = [label for label, _ in pytorch_text_dataset]
            length = len(y_data)
        X_data = LazyIndexable(
            (text for _, text in pytorch_text_dataset), length=length
        )
        y_data = LazyIndexable(y_data, length=length)

        return cls(X_data=X_data, y_data=y_data)

    @classmethod
    def from_array(
        cls, X_array: Union[list, np.ndarray], y_array: Union[list, np.ndarray] = None
    ):
        """Function that create a Dataset from array-like objects, list and numpy.

        Args:
        -----
            X_array (Union[list, np.ndarray]): Array-like containing X_data.
            y_array (Optional[Union[list, np.ndarray]]): Array-like containing the y_data. Default None.

        Returns:
        --------
            Dataset: a Dataset which encasulates X_array and/or y_array.
        """
        if y_array is not None:
            if not isinstance(X_array, (list, np.ndarray)) or not isinstance(
                y_array, (list, np.ndarray)
            ):
                warnings.warn(  # noqa: B028
                    "X_array or y_array are not a list nor a numpy array. The method might not work as expected.",
                    RuntimeWarning,
                )
        else:
            if not isinstance(X_array, (list, np.ndarray)):
                warnings.warn(  # noqa: B028
                    "X_array is not a list nor a numpy array. The method might not work as expected.",
                    RuntimeWarning,
                )

        X_data = LazyIndexable(X_array, length=len(X_array))
        y_data = (
            None if y_array is None else LazyIndexable(y_array, length=len(y_array))
        )

        return cls(X_data=X_data, y_data=y_data)

    def validate(self):
        """Function that checks whether the object is correct or not."""
        try:
            y_data_length = len(self.y_data)
        except TypeError:
            y_data_length = self.y_data.shape[0]
        if self.y_data is not None and len(self) != y_data_length:
            raise ValueError(
                f"X_data and y_data must have equal lenght. X_data has {len(self)} elements and y_data has {y_data_length} elements."
            )
