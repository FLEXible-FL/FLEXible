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
import numpy as np

from flex.data import Dataset, FedDataDistribution, FedDatasetConfig
from flex.datasets import standard_datasets


def federated_emnist(out_dir: str = ".", split="digits", return_test=False):
    train_data, _ = standard_datasets.emnist(out_dir, split=split, include_authors=True)
    config = FedDatasetConfig(
        group_by_label_index=1
    )  # when authoers are included, each label is a tuple (class, writer_id)
    federated_data = FedDataDistribution.from_config(train_data, config)
    if return_test:
        _, test_data = standard_datasets.emnist(
            out_dir, split=split, include_authors=False
        )
        return (federated_data, test_data)
    else:
        return federated_data


def federated_celeba(out_dir: str = ".", return_test=False):
    from torchvision.datasets import CelebA

    class ToNumpy:
        def __call__(self, data):
            if isinstance(data, tuple):  # Label
                return tuple(np.asarray(i) for i in data)
            else:
                return np.asarray(data)  # Images

    dataset = CelebA(
        root=out_dir,
        split="train",
        transform=ToNumpy(),
        target_transform=ToNumpy(),
        target_type=["identity", "attr"],
        download=True,
    )
    config = FedDatasetConfig(group_by_label_index=0)  # identity
    federated_data = FedDataDistribution.from_config_with_torchvision_dataset(
        dataset, config
    )
    if return_test:
        test_ds = CelebA(
            root=out_dir,
            split="test",
            transform=ToNumpy(),
            target_transform=ToNumpy(),
            target_type=["identity", "attr"],
            download=True,
        )
        test_data = Dataset.from_torchvision_dataset(test_ds)
        return (federated_data, test_data)
    return federated_data


def federated_sentiment140(out_dir: str = ".", return_test=False, **kwargs):
    from datasets import load_dataset

    dataset = load_dataset("sentiment140")
    x_labels = ["text"]
    y_labels = ["user", "sentiment"]
    config = FedDatasetConfig(group_by_label_index=0)  # Label "user"
    federated_data = FedDataDistribution.from_config_with_huggingface_dataset(
        dataset["train"], config, x_labels, y_labels
    )
    if return_test:
        test_data = Dataset.from_huggingface_dataset(
            dataset["test"], x_labels, y_labels
        )
        return (federated_data, test_data)
    return federated_data


def federated_shakespeare(out_dir: str = ".", return_test=False):
    train_data, _ = standard_datasets.shakespeare(out_dir, include_actors=True)
    config = FedDatasetConfig(
        group_by_label_index=1
    )  # each label is a pair (class, actor_id)
    federated_data = FedDataDistribution.from_config(train_data, config)
    if return_test:
        _, test_data = standard_datasets.shakespeare(out_dir, include_actors=False)
        return (federated_data, test_data)
    else:
        return federated_data
