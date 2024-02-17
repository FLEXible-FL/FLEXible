from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset


class DefaultVision(VisionDataset):
    def __init__(self, data, transform=None, target_transform=None):
        super().__init__(
            root="",
            transforms=None,
            transform=transform,
            target_transform=target_transform,
        )
        self.data = data

    def __getitem__(self, index: int):
        image, label = self.data[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.data)


class FeatureDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx][0]


class LabelDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self._dataset = dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        return self._dataset[idx][1]
