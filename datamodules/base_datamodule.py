import torch
from typing import Optional
import lightning.pytorch as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from data.dataset import SemiSupervisedDataset, ImageDataset

class BaseDataModule(pl.LightningDataModule):
    """Pytorch Lightning data module class for fine-tuning"""
    def __init__(
        self,
        data,
        batch_size=32,
        num_workers=0,
        transforms=None,
    ):
        """BaseDataModule constructor."""
        super().__init__()
        self.data = data
        self.transforms = transforms
        self.train_batch_size = batch_size
        self.eval_batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):

        if 'U' in self.data:
            self.train_dataset = SemiSupervisedDataset(
                labeled_images=self.data['X'][0], 
                labeled_labels=self.data['X'][1], 
                labeled_h_labels=self.data['X'][2], 
                unlabeled_images=self.data['U'],
                transforms=self.transforms,
                )
        else:
            self.train_dataset = SemiSupervisedDataset(
                labeled_images=self.data['X'][0], 
                labeled_labels=self.data['X'][1], 
                labeled_h_labels=self.data['X'][2], 
                transforms=self.transforms,
                )

        self.val_dataset = SemiSupervisedDataset(
                labeled_images=self.data['X_val'],
                labeled_labels=self.data['Y_val'], 
                labeled_h_labels=self.data['Y_val_h'], 
                unlabeled_images=None,
                transforms=self.transforms,
                )

        self.test_dataset = SemiSupervisedDataset(
                labeled_images=self.data['X_te'], 
                labeled_labels=self.data['Y_te'], 
                labeled_h_labels=self.data['Y_te_h'], 
                unlabeled_images=None,
                transforms=self.transforms,
                )

    def train_dataloader(self):
        """Return training dataset loader."""
        dataloader = torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=True,
                drop_last=True,
            )
        return dataloader

    def val_dataloader(self):
        """Return validation dataset loader."""
        dataloader = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        print(f"Length of val dataloader: {len(dataloader)} batches")
        return dataloader

    def test_dataloader(self):
        """Return test dataset loader."""
        dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        print(f"Length of test dataloader: {len(dataloader)} batches")
        return dataloader
    
    def predict_dataloader(self):
        """Return test dataset loader."""
        dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
        print(f"Length of test dataloader: {len(dataloader)} batches")
        return dataloader
    
# This datamodule is only used to store the images to the disk
class ImageDataModule(pl.LightningDataModule):
    def __init__(self, df, image_size = 224, batch_size=32, num_workers=4, image_reader = None):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_reader = image_reader
        
        self.transform = A.Compose([
                A.Resize(image_size, image_size),
                ToTensorV2()], p=1.)

    def setup(self, stage=None):
        # Load dataset into RAM
        self.dataset = ImageDataset(self.df, transform=self.transform, image_reader=self.image_reader)

    def train_dataloader(self):
        # Create DataLoader for training with multi-CPU support
        return torch.utils.dataDataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True, shuffle = False)