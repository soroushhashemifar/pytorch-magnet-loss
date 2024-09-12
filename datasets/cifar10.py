from torchvision.datasets import CIFAR10
from PIL import Image


class customCIFAR10(CIFAR10):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        print("Size of dataset : " + str(len(self.data)))
        self.read_order = range(0, len(self.data))

    def __getitem__(self, read_index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        index = self.read_order[read_index]

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img, mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.read_order)

    def update_read_order(self, new_order):
        self.read_order = new_order

    def default_read_order(self):
        self.read_order = range(0, len(self.data))