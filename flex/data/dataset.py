import contextlib
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import numpy.typing as npt

from flex.data.lazy_indexable import LazyIndexable


@dataclass(frozen=True)
class Dataset:
    """Class used to represent the dataset from a client in a Federated Learning enviroment.

    Attributes
    ----------
    X_data: numpy.typing.ArrayLike
        A numpy.array containing the data for the client.
    y_data: numpy.typing.ArrayLike
        A numpy.array containing the labels for the training data. Can be None if working
        on an unsupervised learning task. Default None.
    """

    X_data: Union[npt.NDArray, LazyIndexable] = field(init=True)
    y_data: Optional[Union[npt.NDArray, LazyIndexable]] = field(default=None, init=True)

    def __len__(self):
        try:
            return len(self.X_data)
        except TypeError:
            return self.X_data.shape[0]

    def __getitem__(self, index):
        return (
            self.X_data[index],
            self.y_data[index] if self.y_data is not None else None,
        )

    def __iter__(self):
        return zip(
            self.X_data,
            self.y_data if self.y_data is not None else [None] * len(self),
        )

    def to_torchvision_dataset(self, **kwargs):
        """This function transforms a Dataset into a Torchvision dataset object

        Returns:
            torvhcision.datasets.VisionDataset: a torchvision dataset with the contents of datasets. \
                Note that transforms should be pased as arguments.
        """
        from torchvision.datasets import VisionDataset

        class DefaultVision(VisionDataset):
            def __init__(other_self, data, **other_kwargs):
                super().__init__(root="", **other_kwargs)
                other_self.data = data

            def __getitem__(other_self, index: int):
                image, label = other_self.data[index]
                if other_self.transform:
                    image = other_self.transform(image)
                if other_self.target_transform:
                    label = other_self.target_transform(label)
                return image, label

            def __len__(other_self):
                return len(other_self.data)

        return DefaultVision(self, **kwargs)

    def to_tf_dataset(self):
        """This function is an utility to transform a Dataset object to a tensorflow.data.Dataset object

        Returns:
            tensorflow.data.Dataset: tf dataset object instanciated using the contents of a Dataset
        """
        from tensorflow import type_spec_from_value
        from tensorflow.data import Dataset

        return Dataset.from_generator(
            self.__iter__,
            output_signature=(
                type_spec_from_value(self[0][0]),
                type_spec_from_value(self[0][1]),
            ),
        )

    @classmethod
    def from_torchvision_dataset(cls, pytorch_dataset):
        """Function to convert an object from torchvision.datasets.* to a FlexDataObject.

        Args:
            pytorch_dataset (torchvision.datasets.*): a torchvision dataset.

        Returns:
            Dataset: a FlexDataObject which encapsulates the dataset.
        """

        from flex.data.pluggable_datasets import PluggableTorchvision

        if pytorch_dataset.__class__.__name__ not in PluggableTorchvision:
            warnings.warn(
                "The input dataset and arguments are not explicitly supported, therefore they might not work as expected.",
                RuntimeWarning,
            )

        try:
            length = len(pytorch_dataset)
        except TypeError:
            length = None

        X_data = LazyIndexable((x for x, _ in pytorch_dataset), length=length)
        y_data = LazyIndexable((y for _, y in pytorch_dataset), length=length)

        return cls(X_data=X_data, y_data=y_data)

    @classmethod
    def from_tfds_image_dataset(cls, tfds_dataset):
        """Function to convert a dataset from tensorflow_datasets to a FlexDataObject.

        Args:
        ----
            tdfs_dataset (tf.data.Datasets): a tf dataset

        Returns:
        -------
            Dataset: a FlexDataObject which encapsulates the dataset.
        """

        if not isinstance(tfds_dataset, tuple):
            # unbatch if required
            with contextlib.suppress(ValueError):
                tfds_dataset = tfds_dataset.unbatch()
            X_data = LazyIndexable((x for x, _ in tfds_dataset.as_numpy_iterator()))
            y_data = LazyIndexable((y for _, y in tfds_dataset.as_numpy_iterator()))
        else:
            X_data = LazyIndexable(iter(tfds_dataset[0]), length=len(tfds_dataset[0]))
            y_data = LazyIndexable(iter(tfds_dataset[1]), length=len(tfds_dataset[1]))

        return cls(X_data=X_data, y_data=y_data)

    @classmethod
    def from_tfds_text_dataset(cls, tfds_dataset, X_columns=None, label_columns=None):
        """Function to convert a dataset from tensorflow_datasets to a FlexDataObject.

        Args:
            tdfs_dataset (tf.data.Datasets): a tf dataset loaded.
            X_columns (list): List containing the features (input) of the model.
            label_columns (list): List containing the targets of the model.

        Returns:
            Dataset: a FlexDataObject which encapsulates the dataset.
        """
        from tensorflow.python.data.ops.dataset_ops import PrefetchDataset

        if isinstance(tfds_dataset, PrefetchDataset):
            # First case: Users used load func with batch_size != -1 or without indicating the batch_size
            if not isinstance(tfds_dataset, tuple):
                with contextlib.suppress(ValueError):
                    tfds_dataset.unbatch()
            if X_columns is None:
                X_data_generator = iter(tfds_dataset.as_numpy_iterator())
            else:
                X_data_generator = (
                    tuple(map(row.get, X_columns))
                    for row in tfds_dataset.as_numpy_iterator()
                )
            X_data = LazyIndexable(X_data_generator)

            if label_columns is None:
                y_data = None
            else:
                y_data_generator = (
                    tuple(map(row.get, label_columns))
                    for row in tfds_dataset.as_numpy_iterator()
                )
                y_data = LazyIndexable(y_data_generator)
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
    def from_huggingface_dataset(cls, hf_dataset, X_columns, label_columns):
        """Function to conver an arrow dataset from the Datasets package (HuggingFace datasets library)
        to a FlexDataObject.

        Args:
            hf_dataset (datasets.arrow_dataset.Dataset): a dataset from the dataset library
            X_columns (str, list):
            label_columns (str, list): name of the label columns

        Returns:
            Dataset: a FlexDataObject which encapsulates the dataset.
        """
        from flex.data.pluggable_datasets import PluggableHuggingFace

        try:
            if hf_dataset.info.builder_name not in PluggableHuggingFace:
                warnings.warn(
                    "The input dataset and arguments are not explicitly supported, therefore they might not work as expected.",
                    RuntimeWarning,
                )
        except Exception:
            warnings.warn(
                "The input dataset doesn't have the property dataset.info.builder_name, so we can't check if is supported or not. Therefore, it might not work as expected.",
                RuntimeWarning,
            )

        df = hf_dataset.to_pandas()
        X_data = df[X_columns].to_numpy()
        y_data = df[label_columns].to_numpy()
        return cls(X_data=X_data, y_data=y_data)

    @classmethod
    def from_torchtext_dataset(cls, pytorch_text_dataset):
        """Function to convert an object from torchtext.datasets.* to a FlexDataObject.
            It is mandatory that the dataset contains at least the following transform:
            torchtext.transforms.ToTensor()

        Args:
            pytorch_text_dataset (torchtext.datasets.*): a torchtext dataset

        Returns:
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
            length = None
        X_data = LazyIndexable(
            (text for label, text in pytorch_text_dataset), length=length
        )
        y_data = LazyIndexable(
            (label for label, text in pytorch_text_dataset), length=length
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
