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
