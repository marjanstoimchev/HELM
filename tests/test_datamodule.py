"""
Tests for datamodules/base_datamodule.py — BaseDataModule setup and dataloaders.
Requires lightning.
"""
import numpy as np
import torch
import pytest

from tests.conftest import requires_lightning, requires_kornia


def _make_data(n_train=20, n_val=10, n_test=10, n_classes=5, n_classes_h=8,
               with_unlabeled=False):
    """Create synthetic data dict matching DatasetPipeline output."""
    def _imgs(n):
        return np.random.rand(n, 3, 32, 32).astype(np.float32)
    def _labs(n, nc):
        y = np.zeros((n, nc), dtype=np.float32)
        for i in range(n):
            y[i, np.random.choice(nc, 1)] = 1.0
        return y

    x_tr = _imgs(n_train)
    y_tr = _labs(n_train, n_classes)
    y_tr_h = _labs(n_train, n_classes_h)

    data = {
        "X": (x_tr, y_tr, y_tr_h),
        "X_val": _imgs(n_val),
        "Y_val": _labs(n_val, n_classes),
        "Y_val_h": _labs(n_val, n_classes_h),
        "X_te": _imgs(n_test),
        "Y_te": _labs(n_test, n_classes),
        "Y_te_h": _labs(n_test, n_classes_h),
    }
    if with_unlabeled:
        data["U"] = _imgs(n_train * 3)
    return data


@requires_lightning
class TestBaseDataModule:
    def test_setup_supervised(self):
        from datamodules.base_datamodule import BaseDataModule
        data = _make_data(with_unlabeled=False)
        dm = BaseDataModule(data, batch_size=4, num_workers=0)
        dm.setup()
        assert hasattr(dm, "train_dataset")
        assert hasattr(dm, "val_dataset")
        assert hasattr(dm, "test_dataset")

    def test_setup_semisupervised(self):
        from datamodules.base_datamodule import BaseDataModule
        data = _make_data(with_unlabeled=True)
        dm = BaseDataModule(data, batch_size=4, num_workers=0)
        dm.setup()
        assert dm.train_dataset.unlabeled_images is not None

    def test_train_dataloader(self):
        from datamodules.base_datamodule import BaseDataModule
        data = _make_data(n_train=16, with_unlabeled=False)
        dm = BaseDataModule(data, batch_size=4, num_workers=0)
        dm.setup()
        dl = dm.train_dataloader()
        batch = next(iter(dl))
        assert "x" in batch
        assert "one_hot" in batch
        assert batch["x"].shape[0] == 4

    def test_val_dataloader(self):
        from datamodules.base_datamodule import BaseDataModule
        data = _make_data(n_val=8, with_unlabeled=False)
        dm = BaseDataModule(data, batch_size=4, num_workers=0)
        dm.setup()
        dl = dm.val_dataloader()
        batch = next(iter(dl))
        assert "x" in batch

    def test_test_dataloader(self):
        from datamodules.base_datamodule import BaseDataModule
        data = _make_data(n_test=8, with_unlabeled=False)
        dm = BaseDataModule(data, batch_size=4, num_workers=0)
        dm.setup()
        dl = dm.test_dataloader()
        batch = next(iter(dl))
        assert "x" in batch

    def test_predict_dataloader(self):
        from datamodules.base_datamodule import BaseDataModule
        data = _make_data(with_unlabeled=False)
        dm = BaseDataModule(data, batch_size=4, num_workers=0)
        dm.setup()
        dl = dm.predict_dataloader()
        assert dl is not None

    def test_dataloader_drop_last_train(self):
        from datamodules.base_datamodule import BaseDataModule
        data = _make_data(n_train=7, with_unlabeled=False)
        dm = BaseDataModule(data, batch_size=4, num_workers=0)
        dm.setup()
        dl = dm.train_dataloader()
        # With drop_last=True and 7 samples, we should get 1 batch of 4
        batches = list(dl)
        assert len(batches) == 1
        assert batches[0]["x"].shape[0] == 4

    def test_ssl_dataloader_has_unlabeled(self):
        from datamodules.base_datamodule import BaseDataModule
        data = _make_data(n_train=5, with_unlabeled=True)
        dm = BaseDataModule(data, batch_size=4, num_workers=0)
        dm.setup()
        dl = dm.train_dataloader()
        batch = next(iter(dl))
        assert "u" in batch
