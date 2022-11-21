import numpy as np

from flex.data import Dataset, FedDataDistribution, FedDatasetConfig
from flex.datasets.standard_datasets import EMNIST, Shakespeare
from flex.datasets import EncapsulatedDataset


class FederatedEMNIST(EncapsulatedDataset):
    def __init__(self, out_dir: str = ".", split="digits", return_test=False):
        super(FederatedEMNIST, self).__init__(out_dir=out_dir, split=split, return_test=return_test)

    def load_dataset(self):
        train_data, test_data = EMNIST(self.out_dir, split=self.split, include_authors=True)
        config = FedDatasetConfig(
            group_by_label=1
        )  # each label is a pair (class, writer_id)
        federated_data = FedDataDistribution.from_config(train_data, config)
        return (federated_data, test_data) if self.return_test else federated_data


"""
def FederatedEMNIST_f(out_dir: str = ".", split="digits", return_test=False):
    train_data, test_data = EMNIST(out_dir, split=split, include_authors=True)
    config = FedDatasetConfig(
        group_by_label=1
    )  # each label is a pair (class, writer_id)
    federated_data = FedDataDistribution.from_config(train_data, config)
    return (federated_data, test_data) if return_test else federated_data
"""


class FederatedCelebA(EncapsulatedDataset):
    def __init__(self, out_dir: str = ".", return_test=False):
        super(FederatedCelebA, self).__init__(out_dir=out_dir, return_test=return_test)

    def load_dataset(self):
        from torchvision.datasets import CelebA

        class ToNumpy:
            def __call__(self, data):
                if isinstance(data, tuple):  # Label
                    return tuple(np.asarray(i) for i in data)
                else:
                    return np.asarray(data)  # Images

        dataset = CelebA(
            root=self.out_dir,
            split="train",
            transform=ToNumpy(),
            target_transform=ToNumpy(),
            target_type=["identity", "attr"],
            download=True,
        )
        config = FedDatasetConfig(group_by_label=0)  # identity
        federated_data = FedDataDistribution.from_config_with_torchvision_dataset(
            dataset, config
        )
        if self.return_test:
            test_ds = CelebA(
                root=self.out_dir,
                split="test",
                transform=ToNumpy(),
                target_transform=ToNumpy(),
                target_type=["identity", "attr"],
                download=True,
            )
            test_data = Dataset.from_torchvision_dataset(test_ds)
            return (federated_data, test_data)
        return federated_data


"""
def FederatedCelebA_f(out_dir: str = ".", return_test=False):
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
    config = FedDatasetConfig(group_by_label=0)  # identity
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
"""


class FederatedSentiment140(EncapsulatedDataset):
    def __init__(self, out_dir: str = ".", return_test=False):
        super(FederatedSentiment140, self).__init__(out_dir=out_dir, return_test=return_test)

    def load_dataset(self):
        from datasets import load_dataset

        dataset = load_dataset("sentiment140")
        x_labels = "text"
        y_labels = ["user", "sentiment"]
        config = FedDatasetConfig(group_by_label=0)  # Label "user"
        federated_data = FedDataDistribution.from_config_with_huggingface_dataset(
            dataset["train"], config, x_labels, y_labels
        )
        if self.return_test:
            test_data = Dataset.from_huggingface_dataset(
                dataset["test"], x_labels, y_labels
            )
            return (federated_data, test_data)
        return federated_data


"""
def FederatedSentiment140_f(out_dir: str = ".", return_test=False):
    from datasets import load_dataset

    dataset = load_dataset("sentiment140")
    x_labels = "text"
    y_labels = ["user", "sentiment"]
    config = FedDatasetConfig(group_by_label=0)  # Label "user"
    federated_data = FedDataDistribution.from_config_with_huggingface_dataset(
        dataset["train"], config, x_labels, y_labels
    )
    if return_test:
        test_data = Dataset.from_huggingface_dataset(
            dataset["test"], x_labels, y_labels
        )
        return (federated_data, test_data)
    return federated_data
"""


class FederatedShakespeare(EncapsulatedDataset):
    def __init__(self, out_dir: str = ".", return_test=False):
        super(FederatedShakespeare, self).__init__(out_dir=out_dir, return_test=return_test)

    def load_dataset(self):
        train_data, test_data = Shakespeare(self.out_dir, include_actors=True)
        config = FedDatasetConfig(
            group_by_label=1
        )  # each label is a pair (class, actor_id)
        federated_data = FedDataDistribution.from_config(train_data, config)
        return (federated_data, test_data) if self.return_test else federated_data


"""
def FederatedShakespeare_f(out_dir: str = ".", return_test=False):
    train_data, test_data = Shakespeare(out_dir, include_actors=True)
    config = FedDatasetConfig(
        group_by_label=1
    )  # each label is a pair (class, actor_id)
    federated_data = FedDataDistribution.from_config(train_data, config)
    return (federated_data, test_data) if return_test else federated_data
"""
# def FederatedReddit(cls, out_dir: str = ".", split="train"):
#     reddit_files = download_dataset(
#         REDDIT_URL, REDDIT_FILE, REDDIT_MD5, out_dir=out_dir, extract=True, output=True
#     )
#     filtered_files = filter(lambda n: split in n and n.endswith(".json"), reddit_files)
#     flex_dataset = FedDataset()
#     for f in tqdm(filtered_files):
#         with open(f) as json_file:
#             train_data = json.load(json_file)
#         for user_id in train_data['users']:
#             node_ds = train_data['user_data'][user_id]
#             y_data = [(y["target_tokens"], y["count_tokens"]) for y in node_ds['y']]
#             x_data = node_ds['x']
#             flex_dataset[user_id] = Dataset(x_data, y_data)
#     return flex_dataset
