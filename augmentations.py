import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import kornia as K
from kornia import image_to_tensor

class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, mode) -> None:
        super().__init__()
        
        if mode == 'weak':
            self.transforms = nn.Sequential(
                K.augmentation.RandomHorizontalFlip(p=0.5),
                K.augmentation.RandomVerticalFlip(p=0.5),
                K.augmentation.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.25),
            )
        elif mode == 'strong':
            self.transforms = nn.Sequential(
                K.augmentation.RandomHorizontalFlip(p=0.5),
                K.augmentation.RandomVerticalFlip(p=0.5),
                K.augmentation.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
                K.augmentation.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10, p=0.8),
                K.augmentation.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.75, 1.33), p=0.8),
                K.augmentation.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),
                K.augmentation.RandomErasing(p=0.5),
            )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        x_out = self.transforms(x)  # BxCxHxW
        return x_out
    
class Preprocess(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""
    def __init__(self, image_size = 224) -> None:
        self.image_size = image_size
        super().__init__()
        self.resize = K.augmentation.Resize((self.image_size, self.image_size), keepdim=True, antialias=True)

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) -> Tensor:
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_out: Tensor = image_to_tensor(x_tmp, keepdim=True).float() / 255.0 # CxHxW
        x_out: Tensor = self.resize(x_out)
        return x_out