from typing import Optional, Union, List, Any

from PIL import Image
from pytorch_lightning.trainer.supporters import CombinedLoader
from torch.utils.data import random_split, Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.datasets import MNIST
import torch

from dataset.base import DataModuleBase


class CustomDataset(Dataset):

    def __init__(self, data, targets, transform=None):
        super(CustomDataset, self).__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class MNISTDataModule(DataModuleBase):

    def __init__(self, *args, **kwargs):
        # DataModuleBase.convert_arguments(kwargs=kwargs)
        super(MNISTDataModule, self).__init__(*args, **kwargs)
        self.train_id_dataset = None
        self.train_ood_dataset = None
        self.val_id_dataset = None
        self.val_ood_dataset = None

        # Transform
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.dims = (1, 28, 28)

    def prepare_data(self, *args, **kwargs):
        # Download
        MNIST(root=self.data_dir, train=True, download=True)
        MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        stage = stage

        train_val_ratio = self.hparams["train_val_ratio"]
        train_id_true_target_list = self.hparams["train_id_targets"]
        train_ood_true_target_list = self.hparams["train_ood_targets"]
        test_ood_true_target_list = self.hparams["test_ood_targets"]

        # Set stage
        if stage == "fit" or stage == "whole":
            stage = "fit"
            mnist_entire = MNIST(root=self.data_dir, train=True, transform=self.transform)
            target_list = mnist_entire.targets

            train_id_targets_bool = torch.full((target_list.shape[0],), False)
            train_ood_targets_bool = torch.full((target_list.shape[0],), False)

            # Train id targets
            for train_id_true_target in train_id_true_target_list:
                train_id_targets_bool += (target_list == train_id_true_target)

            # Train ood targets
            for train_ood_true_target in train_ood_true_target_list:
                train_ood_targets_bool += (target_list == train_ood_true_target)

            train_id_data = mnist_entire.data[train_id_targets_bool]
            train_id_targets = target_list[train_id_targets_bool]
            train_ood_data = mnist_entire.data[train_ood_targets_bool]
            train_ood_targets = target_list[train_ood_targets_bool]

            # Split train and val
            mnist_id_entire = CustomDataset(data=train_id_data, targets=train_id_targets, transform=self.transform)
            id_train_size = int(len(mnist_id_entire) * train_val_ratio)
            id_val_size = len(mnist_id_entire) - id_train_size

            mnist_ood_entire = CustomDataset(data=train_ood_data, targets=train_ood_targets, transform=self.transform)
            ood_train_size = int(len(mnist_ood_entire) * train_val_ratio)
            ood_val_size = len(mnist_ood_entire) - ood_train_size

            # Set dataset
            self.train_id_dataset, self.val_id_dataset = random_split(mnist_id_entire, [id_train_size, id_val_size])
            self.train_ood_dataset, self.val_ood_dataset = random_split(mnist_ood_entire, [ood_train_size, ood_val_size])

        if stage == "test" or stage == "whole":
            stage = "test"
            # self.test_dataset = MNIST(root=self.data_dir, train=False, transform=self.transform)
            mnist_test_entire = MNIST(root=self.data_dir, train=False, transform=self.transform)
            target_list = mnist_test_entire.targets

            test_targets_bool = torch.full((target_list.shape[0],), False)

            # Test targets
            for train_id_true_target in train_id_true_target_list:
                test_targets_bool += (target_list == train_id_true_target)

            for test_ood_true_target in test_ood_true_target_list:
                test_targets_bool += (target_list == test_ood_true_target)

            test_data = mnist_test_entire.data[test_targets_bool]
            test_targets = target_list[test_targets_bool]

            # Set dataset
            self.test_dataset = CustomDataset(data=test_data, targets=test_targets, transform=self.transform)

    def train_id_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_id_dataset,
            num_workers=self.datamodule_params.num_workers,
            pin_memory=self.datamodule_params.pin_memory,
            batch_size=self.datamodule_params.batch_size
        )

    def val_id_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_id_dataset,
            num_workers=self.datamodule_params.num_workers,
            pin_memory=self.datamodule_params.pin_memory,
            batch_size=self.datamodule_params.batch_size
        )

    def train_ood_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_ood_dataset,
            num_workers=self.datamodule_params.num_workers,
            pin_memory=self.datamodule_params.pin_memory,
            batch_size=self.datamodule_params.batch_size
        )

    def val_ood_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_ood_dataset,
            num_workers=self.datamodule_params.num_workers,
            pin_memory=self.datamodule_params.pin_memory,
            batch_size=self.datamodule_params.batch_size
        )

    def train_occ_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_id_dataset,
            num_workers=self.datamodule_params.num_workers,
            pin_memory=self.datamodule_params.pin_memory,
            batch_size=self.datamodule_params.batch_size
        )

    def val_occ_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=ConcatDataset(datasets=[self.val_id_dataset, self.val_ood_dataset]),
            num_workers=self.datamodule_params.num_workers,
            pin_memory=self.datamodule_params.pin_memory,
            batch_size=self.datamodule_params.batch_size
        )

    def train_dataloader(self):
        return CombinedLoader(
            loaders=dict(
                train_id=self.train_id_dataloader(),
                train_ood=self.train_ood_dataloader()
            ),
            mode="max_size_cycle"
        )

    def val_dataloader(self):
        return CombinedLoader(
            loaders=dict(
                val_id=self.val_id_dataloader(),
                val_ood=self.val_ood_dataloader()
            ),
            mode="max_size_cycle"
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            num_workers=self.datamodule_params.num_workers,
            pin_memory=self.datamodule_params.pin_memory,
            batch_size=self.datamodule_params.batch_size
        )

    # def convert_label(self, id_classes_idx_list=[1, 3, 5, 7, 9], train_ood_classes_idx_list=[0, 2, 4], test_ood_classes_idx_list=[6, 8]):
    #     if id_classes_idx_list:
    #         id_train_dataset = None
    #         id_val_dataset = None
    #         id_test_dataset = None
    #
    #         for data in self.train_dataset:
    #             if data[1] in id_classes_idx_list:
    #
    #         pass
    #     if train_ood_classes_idx_list:
    #         pass
    #     if test_ood_classes_idx_list:
    #         pass
    #     self.train_dataset
    #     self.val_dataset


class MNISTCustomDataset(MNIST):

    def __init__(self, *args, **kwargs):
        super(MNISTCustomDataset, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        # Convert target

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target
