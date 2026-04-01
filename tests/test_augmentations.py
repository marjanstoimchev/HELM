"""
Tests for augmentations.py — DataAugmentation and Preprocess modules.
Requires kornia.
"""
import numpy as np
import torch
import pytest

from tests.conftest import requires_kornia


@requires_kornia
class TestDataAugmentation:
    def test_weak_augmentation_output_shape(self):
        from augmentations import DataAugmentation
        aug = DataAugmentation(mode="weak")
        x = torch.randn(4, 3, 224, 224)
        out = aug(x)
        assert out.shape == x.shape

    def test_strong_augmentation_output_shape(self):
        from augmentations import DataAugmentation
        aug = DataAugmentation(mode="strong")
        x = torch.randn(4, 3, 224, 224)
        out = aug(x)
        assert out.shape == x.shape

    def test_no_gradients_computed(self):
        from augmentations import DataAugmentation
        aug = DataAugmentation(mode="weak")
        x = torch.randn(4, 3, 224, 224, requires_grad=True)
        out = aug(x)
        assert not out.requires_grad


@requires_kornia
class TestPreprocess:
    def test_output_shape(self):
        from augmentations import Preprocess
        prep = Preprocess(image_size=224)
        # Preprocess expects HxWxC numpy-like input
        x = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out = prep(x)
        assert out.shape[-2:] == (224, 224)

    def test_output_normalized(self):
        from augmentations import Preprocess
        prep = Preprocess(image_size=224)
        x = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        out = prep(x)
        assert out.max() <= 1.0
        assert out.min() >= 0.0

    def test_output_is_tensor(self):
        from augmentations import Preprocess
        prep = Preprocess(image_size=224)
        x = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        out = prep(x)
        assert isinstance(out, torch.Tensor)

    def test_resize_from_different_size(self):
        from augmentations import Preprocess
        prep = Preprocess(image_size=224)
        x = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        out = prep(x)
        assert out.shape[-2:] == (224, 224)
