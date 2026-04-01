"""
Tests for data/dataset.py — SemiSupervisedDataset, ImageDataset.
"""
import numpy as np
import pytest
import torch

from data.dataset import SemiSupervisedDataset


# ═══════════════════════════════════════════════════════════════════════════
# 1.  SemiSupervisedDataset — supervised mode (no unlabeled)
# ═══════════════════════════════════════════════════════════════════════════

class TestSemiSupervisedDatasetSupervised:
    def _make(self, n=20, n_classes=5):
        images = np.random.rand(n, 3, 32, 32).astype(np.float32)
        labels = np.zeros((n, n_classes), dtype=np.float32)
        h_labels = np.zeros((n, n_classes + 3), dtype=np.float32)
        for i in range(n):
            active = np.random.choice(n_classes, size=1)
            labels[i, active] = 1.0
            h_labels[i, active] = 1.0
        return images, labels, h_labels

    def test_length(self):
        imgs, labs, hlabs = self._make(20)
        ds = SemiSupervisedDataset(imgs, labs, hlabs)
        assert len(ds) == 20

    def test_getitem_keys(self):
        imgs, labs, hlabs = self._make(10)
        ds = SemiSupervisedDataset(imgs, labs, hlabs)
        item = ds[0]
        assert "x" in item
        assert "one_hot" in item
        assert "h_one_hot" in item
        assert "u" not in item

    def test_getitem_shapes(self):
        imgs, labs, hlabs = self._make(10, n_classes=5)
        ds = SemiSupervisedDataset(imgs, labs, hlabs)
        item = ds[0]
        # x is HxWxC after moveaxis
        assert item["x"].shape == (32, 32, 3)
        assert item["one_hot"].shape == (5,)
        assert item["h_one_hot"].shape == (8,)

    def test_labels_are_tensors(self):
        imgs, labs, hlabs = self._make(10)
        ds = SemiSupervisedDataset(imgs, labs, hlabs)
        item = ds[0]
        assert isinstance(item["one_hot"], torch.Tensor)
        assert isinstance(item["h_one_hot"], torch.Tensor)


# ═══════════════════════════════════════════════════════════════════════════
# 2.  SemiSupervisedDataset — semi-supervised mode (with unlabeled)
# ═══════════════════════════════════════════════════════════════════════════

class TestSemiSupervisedDatasetSSL:
    def _make(self, n_labeled=10, n_unlabeled=30, n_classes=5):
        imgs_l = np.random.rand(n_labeled, 3, 32, 32).astype(np.float32)
        labs = np.zeros((n_labeled, n_classes), dtype=np.float32)
        hlabs = np.zeros((n_labeled, n_classes + 3), dtype=np.float32)
        imgs_u = np.random.rand(n_unlabeled, 3, 32, 32).astype(np.float32)
        return imgs_l, labs, hlabs, imgs_u

    def test_length_matches_unlabeled(self):
        imgs_l, labs, hlabs, imgs_u = self._make(10, 30)
        ds = SemiSupervisedDataset(imgs_l, labs, hlabs, unlabeled_images=imgs_u)
        # Labeled is upsampled to match unlabeled, so length = n_unlabeled
        assert len(ds) == 30

    def test_getitem_has_unlabeled(self):
        imgs_l, labs, hlabs, imgs_u = self._make(10, 30)
        ds = SemiSupervisedDataset(imgs_l, labs, hlabs, unlabeled_images=imgs_u)
        item = ds[0]
        assert "u" in item

    def test_labeled_upsampled(self):
        imgs_l, labs, hlabs, imgs_u = self._make(5, 20)
        ds = SemiSupervisedDataset(imgs_l, labs, hlabs, unlabeled_images=imgs_u)
        # After upsampling, labeled_images should have same size as unlabeled
        assert len(ds.labeled_images) == 20

    def test_no_data_leakage_between_labeled_and_unlabeled(self):
        """Labeled and unlabeled images should come from separate sources."""
        imgs_l = np.ones((5, 3, 32, 32), dtype=np.float32) * 0.1
        labs = np.zeros((5, 3), dtype=np.float32)
        hlabs = np.zeros((5, 5), dtype=np.float32)
        imgs_u = np.ones((10, 3, 32, 32), dtype=np.float32) * 0.9
        ds = SemiSupervisedDataset(imgs_l, labs, hlabs, unlabeled_images=imgs_u)
        # Unlabeled images should be all ~0.9
        for i in range(len(ds)):
            item = ds[i]
            assert item["u"].mean() == pytest.approx(0.9, abs=0.01)
