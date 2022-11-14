import contextlib
import json
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt
from cardinality import count
from lazyarray import larray
from scipy.io import loadmat

from flex.data.utils import (
    EMNIST_DIGITS_FILE,
    EMNIST_DIGITS_MD5,
    EMNIST_DIGITS_URL,
    EMNIST_LETTERS_FILE,
    EMNIST_LETTERS_MD5,
    EMNIST_LETTERS_URL,
    SHAKESPEARE_FILE,
    SHAKESPEARE_MD5,
    SHAKESPEARE_URL,
    download_dataset,
)


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

    X_data: npt.NDArray = field(init=True)
    y_data: Optional[npt.NDArray] = field(default=None, init=True)

    def __len__(self):
        try:
            return len(self.X_data)
        except TypeError:
            return self.X_data.shape[0]

    def __getitem__(self, pos):
        if self.y_data is None:
            return Dataset(self.X_data[pos], None)
        else:
            return Dataset(
                self.X_data[pos],
                self.y_data[pos[0]] if isinstance(pos, tuple) else self.y_data[pos],
            )

    def __iter__(self):
        return zip(
            self.X_data,
            self.y_data if self.y_data is not None else [None] * len(self),
        )

    @classmethod
    def Shakespeare(cls, out_dir: str = ".", include_actors=False):
        shakespeare_files = download_dataset(
            SHAKESPEARE_URL,
            SHAKESPEARE_FILE,
            SHAKESPEARE_MD5,
            out_dir=out_dir,
            extract=True,
            output=True,
        )
        train_files = filter(
            lambda n: "train" in n and n.endswith(".json"), shakespeare_files
        )
        train_x = []
        train_y = []
        for f in train_files:
            with open(f) as json_file:
                train_data = json.load(json_file)
            for user_id in train_data["users"]:
                node_ds = train_data["user_data"][user_id]
                if include_actors:
                    train_y += [(y, user_id) for y in node_ds["y"]]
                else:
                    train_y += node_ds["y"]
                train_x += node_ds["x"]
        test_files = filter(
            lambda n: "test" in n and n.endswith(".json"), shakespeare_files
        )
        test_x = []
        test_y = []
        for f in test_files:
            with open(f) as json_file:
                test_data = json.load(json_file)
            for user_id in test_data["users"]:
                node_ds = test_data["user_data"][user_id]
                if include_actors:
                    test_y += [(y, user_id) for y in node_ds["y"]]
                else:
                    test_y += node_ds["y"]
                test_x += node_ds["x"]

        return Dataset(train_x, train_y), Dataset(test_x, test_y)

    @classmethod
    def EMNIST(cls, out_dir: str = ".", split="digits", include_authors=False):
        if split == "digits":
            url, filename, md5 = (
                EMNIST_DIGITS_URL,
                EMNIST_DIGITS_FILE,
                EMNIST_DIGITS_MD5,
            )
        elif split == "letters":
            url, filename, md5 = (
                EMNIST_LETTERS_URL,
                EMNIST_LETTERS_FILE,
                EMNIST_LETTERS_MD5,
            )
        else:
            raise ValueError(
                f"Unknown split: {split}. Available splits are 'digits' and 'letters'."
            )
        mnist_files = download_dataset(
            url, filename, md5, out_dir=out_dir, extract=False, output=True
        )
        dataset = loadmat(mnist_files)["dataset"]
        train_writers = dataset["train"][0, 0]["writers"][0, 0]
        train_data = np.reshape(
            dataset["train"][0, 0]["images"][0, 0], (-1, 28, 28), order="F"
        )
        train_labels = np.squeeze(dataset["train"][0, 0]["labels"][0, 0])
        if include_authors:
            train_labels = [
                (label, train_writers[i][0]) for i, label in enumerate(train_labels)
            ]

        test_writers = dataset["test"][0, 0]["writers"][0, 0]
        test_data = np.reshape(
            dataset["test"][0, 0]["images"][0, 0], (-1, 28, 28), order="F"
        )
        test_labels = np.squeeze(dataset["test"][0, 0]["labels"][0, 0])
        if include_authors:
            test_labels = [
                (label, test_writers[i][0]) for i, label in enumerate(test_labels)
            ]
        train_data_object = Dataset(X_data=np.asarray(train_data), y_data=train_labels)
        test_data_object = Dataset(X_data=np.asarray(test_data), y_data=test_labels)
        return train_data_object, test_data_object

    @classmethod
    def from_torchvision_dataset(cls, pytorch_dataset):
        """Function to convert an object from torchvision.datasets.* to a FlexDataObject.

        Args:
            pytorch_dataset (torchvision.datasets.*): a torchvision dataset.

        Returns:
            Dataset: a FlexDataObject which encapsulates the dataset.
        """
        from torchvision.datasets import ImageFolder

        from flex.data.pluggable_datasets import PluggableTorchvision

        if pytorch_dataset.__class__.__name__ not in PluggableTorchvision:
            warnings.warn(
                "The input dataset and arguments are not explicitly supported, therefore they might not work as expected.",
                RuntimeWarning,
            )

        length = count(pytorch_dataset)
        if length > 60_000 or isinstance(
            pytorch_dataset, ImageFolder
        ):  # skip loading dataset in memory

            def lazy_1d_index(indices, ds, extra_dim=1):
                try:
                    iter(indices)
                except TypeError:  # not iterable
                    return ds[indices][extra_dim]
                else:  # iterable
                    return larray(
                        lambda a: lazy_1d_index(indices[a], ds, extra_dim),
                        shape=(len(indices),),
                    )

            X_data = larray(
                lambda a: lazy_1d_index(a, pytorch_dataset, extra_dim=0),
                shape=(length,),
            )
            dtype = type(pytorch_dataset[0][1])
            y_data = np.fromiter(
                (y for _, y in pytorch_dataset), dtype=dtype, count=length
            )
        else:
            X_data, y_data = [], []
            for x, y in pytorch_dataset:
                X_data.append(x)
                y_data.append(y)
            X_data = np.asarray(X_data)
            y_data = np.asarray(y_data)
        return cls(X_data=X_data, y_data=y_data)

    @classmethod
    def from_tfds_image_dataset(cls, tfds_dataset):
        """Function to convert a dataset from tensorflow_datasets to a FlexDataObject.

        Args:
            tdfs_dataset (tf.data.Datasets): a tf dataset

        Returns:
            Dataset: a FlexDataObject which encapsulates the dataset.
        """
        from tensorflow_datasets import as_numpy

        # unbatch if possible
        if not isinstance(tfds_dataset, tuple):
            with contextlib.suppress(ValueError):
                tfds_dataset = tfds_dataset.unbatch()
            X_data, y_data = [], []
            for x, y in tfds_dataset.as_numpy_iterator():
                X_data.append(x)
                y_data.append(y)
        else:
            X_data, y_data = as_numpy(tfds_dataset)

        return cls(X_data=np.asarray(X_data), y_data=np.asarray(y_data))

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
        import pandas as pd
        from tensorflow.python.data.ops.dataset_ops import PrefetchDataset
        from tensorflow_datasets import as_dataframe

        if isinstance(tfds_dataset, PrefetchDataset):
            # First case: Users used load func with batch_size != -1 or without indicating the batch_size
            if not isinstance(tfds_dataset, tuple):
                with contextlib.suppress(ValueError):
                    tfds_dataset.unbatch()
            X_data = as_dataframe(tfds_dataset)[X_columns].to_numpy()
            y_data = as_dataframe(tfds_dataset)[label_columns].to_numpy()
        else:  # User used batch_size=-1 when using the load function
            X_data = pd.DataFrame.from_dict(
                {col: tfds_dataset[col].numpy() for col in X_columns}
            ).to_numpy()
            y_data = pd.DataFrame.from_dict(
                {col: tfds_dataset[col].numpy() for col in label_columns}
            ).to_numpy()
        # if len(y_data.shape) == 2 and y_data.shape[1] == 1:
        y_data = np.squeeze(y_data)  # .reshape((len(y_data),))
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
        import numpy as np
        from torch.utils.data import DataLoader

        from flex.data.pluggable_datasets import PluggableTorchtext

        if pytorch_text_dataset.__class__.__name__ not in PluggableTorchtext:
            warnings.warn(
                "The input dataset and arguments are not explicitly supported, therefore they might not work as expected.",
                RuntimeWarning,
            )

        loader = DataLoader(pytorch_text_dataset, batch_size=1)
        X_data, y_data = [], []
        for label, text in loader:
            y_data.append(label.numpy()[0])
            X_data.append(text[0])
        X_data = np.asarray(X_data)
        y_data = np.asarray(y_data)

        return cls(X_data=X_data, y_data=y_data)

    def validate(self):
        """Function that checks whether the object is correct or not."""
        if self.y_data is not None and len(self) != len(self.y_data):
            raise ValueError(
                f"X_data and y_data must have equal lenght. X_data has {len(self)} elements and y_data has {len(self.y_data)} elements."
            )