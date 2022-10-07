from dataclasses import dataclass, field
from typing import Optional

import numpy.typing as npt


@dataclass(frozen=True)
class FlexDataObject:
    """Class used to represent the dataset from a client in a Federated Learning enviroment.

    Attributes
    ----------
    _X_data: numpy.typing.ArrayLike
        A numpy.array containing the data for the client.
    _y_data: numpy.typing.ArrayLike
        A numpy.array containing the labels for the training data. Can be None if working
        on an unsupervised learning task. Default None.
    """

    X_data: npt.NDArray = field(init=True)
    y_data: Optional[npt.NDArray] = field(default=None, init=True)

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, pos):
        if self.y_data is None:
            return FlexDataObject(self.X_data[pos], None)
        else:
            return FlexDataObject(
                self.X_data[pos],
                self.y_data[pos[0]] if isinstance(pos, tuple) else self.y_data[pos],
            )

    def __iter__(self):
        return zip(
            self.X_data,
            self.y_data if self.y_data is not None else [None] * len(self.X_data),
        )

    @classmethod
    def from_torchvision_dataset(cls, pytorch_dataset):
        """Function to convert an object from torchvision.datasets.* to a FlexDataObject.
            It is mandatory that the dataset contains at least the following transform:
            torchvision.transforms.ToTensor()

        Args:
            pytorch_dataset (torchvision.datasets.VisionDataset): a torchvision dataset
            that inherits from torchvision.datasets.VisionDataset.

        Returns:
            FlexDataObject: a FlexDataObject which encapsulates the dataset.
        """
        from torch.utils.data import DataLoader

        loader = DataLoader(pytorch_dataset, batch_size=len(pytorch_dataset))
        try:
            X_data = next(iter(loader))[0].numpy()
            y_data = next(iter(loader))[1].numpy()
        except TypeError as e:
            raise ValueError(
                "When loading a torchvision dataset, provide it with at least \
                torchvision.transforms.ToTensor() in the tranform field."
            ) from e

        return cls(X_data=X_data, y_data=y_data)

    @classmethod
    def from_tfds_dataset(cls, tdfs_dataset):
        """Function to convert a dataset from tensorflow_datasets to a FlexDataObject.
            It is mandatory that the dataset is loaded with batch_size=-1 in tensorflow_datasets.load function.

        Args:
            tdfs_dataset (tf.data.Datasets): a tf dataset

        Returns:
            FlexDataObject: a FlexDataObject which encapsulates the dataset.
        """
        from tensorflow.python.data.ops.dataset_ops import PrefetchDataset
        from tensorflow_datasets import as_numpy

        if isinstance(tdfs_dataset, PrefetchDataset):
            raise ValueError(
                "When loading a tensorflow_dataset, provide it with option batch_size=-1 in tensorflow_datasets.load function."
            )
        if isinstance(tdfs_dataset, list) and len(tdfs_dataset) == 1:
            tdfs_dataset = tdfs_dataset[0]
        X_data, y_data = as_numpy(tdfs_dataset)
        return cls(X_data=X_data, y_data=y_data)

    @classmethod
    def from_huggingface_dataset(cls, hf_dataset, X_columns, label_column):
        """Function to conver a dataset from the Datasets package (HuggingFace datasets library)
        to a FlexDataObject.

        Args:
            hf_dataset (datasets.arrow_dataset.Dataset): a dataset from the dataset library
            X_columns (str, list):
            label_column (str): name of the label column

        Returns:
            FlexDataObject: a FlexDataObject which encapsulates the dataset.
        """
        from datasets.arrow_dataset import Dataset

        if not isinstance(hf_dataset, Dataset):
            raise ValueError(
                "When loading a huggingface_dataset, provide it with the default format: datasets.arrow_dataset.Dataset."
            )
        df = hf_dataset.to_pandas()
        X_data = df[X_columns].to_numpy()
        y_data = df[label_column].to_numpy()
        return cls(X_data=X_data, y_data=y_data)

    @classmethod
    def from_torchtext_dataset(cls, pytorch_text_dataset):
        """Function to convert an object from torchtext.datasets.* to a FlexDataObject.
            It is mandatory that the dataset contains at least the following transform:
            torchtext.transforms.ToTensor()

        Args:
            pytorch_dataset (torchtext.datasets.VisionDataset): a torchtext dataset
            that inherits from torchtext.datasets.VisionDataset.

        Returns:
            FlexDataObject: a FlexDataObject which encapsulates the dataset.
        """
        import numpy as np
        from torch.utils.data import DataLoader, Dataset

        if not isinstance(pytorch_text_dataset, Dataset):
            raise ValueError(
                "When loading a pytorch text dataset, it must be an instance of torch.utils.data.Dataset."
            )
        loader = DataLoader(pytorch_text_dataset, batch_size=1)
        X_data, y_data = [], []
        for label, text in loader:
            y_data.append(label.numpy()[0])
            X_data.append(text[0])
        X_data = np.array(X_data)
        y_data = np.array(y_data)

        return cls(X_data=X_data, y_data=y_data)

    def validate(self):
        """Function that checks whether the object is correct or not."""
        if self.y_data is not None and len(self.X_data) != len(self.y_data):
            raise ValueError(
                f"X_data and y_data must have equal lenght. X_data has {len(self.X_data)} elements and y_data has {len(self.y_data)} elements."
            )
        if self.y_data is not None and self.y_data.ndim > 1:
            raise ValueError(
                "y_data is multidimensional and we only support unidimensional labels."
            )
