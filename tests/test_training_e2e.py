"""
End-to-end training tests — run 1-2 epochs of training with synthetic data
on CPU/single-GPU to verify the full pipeline works.

These are slow tests (marked with pytest.mark.slow).
Requires the full ML stack.
"""
import os
import numpy as np
import torch
import pytest

from tests.conftest import (
    requires_full_stack, requires_cuda, requires_multi_gpu,
    PROJECT_ROOT,
)


def _make_synthetic_data(n_train=24, n_val=8, n_test=8,
                         n_classes=5, n_classes_h=8,
                         with_unlabeled=False):
    """Synthetic data dict matching pipeline output."""
    def _imgs(n):
        return np.random.rand(n, 3, 224, 224).astype(np.float32)
    def _labs(n, nc):
        y = np.zeros((n, nc), dtype=np.float32)
        for i in range(n):
            y[i, np.random.choice(nc, size=np.random.randint(1, 3), replace=False)] = 1.0
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
        data["U"] = _imgs(n_train * 2)
    return data


def _make_edge_index():
    return torch.tensor([[5, 5, 6, 6, 7],
                         [0, 1, 2, 3, 4]], dtype=torch.long)


def _run_training(trainer_cls, data, num_classes_h, num_leaves, edge_index,
                  learning_task, accelerator="gpu", devices=None, max_epochs=2):
    """Helper to run a short training loop on GPU 3."""
    import lightning as L
    from models.model import h_deit_base_embedding
    from augmentations import Preprocess
    from datamodules.base_datamodule import BaseDataModule
    from utils.utils import Dotdict

    if devices is None:
        devices = [3]

    config = Dotdict({
        "training": {
            "lr": 1e-4, "head_lr": 1e-4, "max_lr": 3e-4,
            "apply_scheduler": False, "epochs": max_epochs,
            "min_epochs": 1, "patience": 5, "lr_schedule_patience": 5,
            "accumulate_grad_batches": 1, "deterministic": True,
            "log_every_n_steps": 1,
        },
        "dataset": {
            "name": "test", "folder_name": "test", "num_classes": num_leaves,
        },
    })

    transforms = Preprocess()
    dm = BaseDataModule(data, batch_size=4, num_workers=0, transforms=transforms)

    backbone = h_deit_base_embedding(num_classes=num_classes_h, pretrained=False)
    model = trainer_cls(config, backbone, num_leaves=num_leaves,
                        learning_task=learning_task, edge_index=edge_index)

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy="auto",
        max_epochs=max_epochs,
        min_epochs=1,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )
    trainer.fit(model, datamodule=dm)
    return trainer, model, dm


# ═══════════════════════════════════════════════════════════════════════════
# Single GPU training tests — all methods on GPU 3
# ═══════════════════════════════════════════════════════════════════════════

@requires_full_stack
@requires_cuda
@pytest.mark.slow
class TestTrainingSingleGPU:
    """End-to-end training on a single GPU with synthetic data."""

    def test_supervised_mlc(self):
        from trainers.supervised import SupervisedModel
        data = _make_synthetic_data(n_classes=5, n_classes_h=5, with_unlabeled=False)
        trainer, model, dm = _run_training(
            SupervisedModel, data, num_classes_h=5, num_leaves=5,
            edge_index=None, learning_task="mlc", max_epochs=1,
        )
        assert trainer.current_epoch == 1

    def test_supervised_hmlc(self):
        from trainers.supervised import SupervisedModel
        data = _make_synthetic_data(n_classes=5, n_classes_h=8, with_unlabeled=False)
        trainer, model, dm = _run_training(
            SupervisedModel, data, num_classes_h=8, num_leaves=5,
            edge_index=None, learning_task="hmlc", max_epochs=1,
        )
        assert trainer.current_epoch == 1

    def test_ssl_byol(self):
        from trainers.ssl_byol import SemiSupervisedBYOLModel
        data = _make_synthetic_data(n_classes=5, n_classes_h=8, with_unlabeled=True)
        trainer, model, dm = _run_training(
            SemiSupervisedBYOLModel, data, num_classes_h=8, num_leaves=5,
            edge_index=None, learning_task="hmlc", max_epochs=1,
        )
        assert trainer.current_epoch == 1

    def test_ssl_graph(self):
        from trainers.ssl_graph import GraphBasedModel
        data = _make_synthetic_data(n_classes=5, n_classes_h=8, with_unlabeled=True)
        ei = _make_edge_index()
        trainer, model, dm = _run_training(
            GraphBasedModel, data, num_classes_h=8, num_leaves=5,
            edge_index=ei, learning_task="hmlc", max_epochs=1,
        )
        assert trainer.current_epoch == 1

    def test_ssl_graph_byol(self):
        from trainers.ssl_graph_byol import SemiSupervisedGraphBYOLModel
        data = _make_synthetic_data(n_classes=5, n_classes_h=8, with_unlabeled=True)
        ei = _make_edge_index()
        trainer, model, dm = _run_training(
            SemiSupervisedGraphBYOLModel, data, num_classes_h=8, num_leaves=5,
            edge_index=ei, learning_task="hmlc", max_epochs=1,
        )
        assert trainer.current_epoch == 1

    def test_prediction_after_training(self):
        from trainers.supervised import SupervisedModel
        from utils.utils import predict, calculate_metrics
        data = _make_synthetic_data(n_classes=5, n_classes_h=5, with_unlabeled=False)
        trainer, model, dm = _run_training(
            SupervisedModel, data, num_classes_h=5, num_leaves=5,
            edge_index=None, learning_task="mlc", max_epochs=1,
        )
        Y = predict(trainer, model, dm)
        assert "y_true" in Y
        assert "y_pred" in Y
        assert "y_scores" in Y
        assert Y["y_true"].shape == Y["y_pred"].shape

        df = calculate_metrics(Y)
        assert "micro f1" in df.index


# ═══════════════════════════════════════════════════════════════════════════
# Prediction test (reuses trainer from fit)
# ═══════════════════════════════════════════════════════════════════════════

@requires_full_stack
@requires_cuda
@pytest.mark.slow
class TestPrediction:
    """Test prediction + metrics pipeline after training."""

    def test_prediction_and_metrics(self):
        from trainers.supervised import SupervisedModel
        from utils.utils import predict, calculate_metrics
        data = _make_synthetic_data(n_classes=5, n_classes_h=5, with_unlabeled=False)
        trainer, model, dm = _run_training(
            SupervisedModel, data, num_classes_h=5, num_leaves=5,
            edge_index=None, learning_task="mlc", max_epochs=1,
        )
        Y = predict(trainer, model, dm)
        assert "y_true" in Y
        assert "y_pred" in Y
        assert "y_scores" in Y
        assert Y["y_true"].shape == Y["y_pred"].shape

        df = calculate_metrics(Y)
        assert "micro f1" in df.index


# ═══════════════════════════════════════════════════════════════════════════
# Multi-GPU training tests (DDP)
# ═══════════════════════════════════════════════════════════════════════════

@requires_full_stack
@requires_multi_gpu
@pytest.mark.slow
class TestTrainingMultiGPU:
    """End-to-end training on multiple GPUs with DDP."""

    def test_supervised_ddp(self):
        import lightning as L
        from trainers.supervised import SupervisedModel
        from models.model import h_deit_base_embedding
        from augmentations import Preprocess
        from datamodules.base_datamodule import BaseDataModule
        from utils.utils import Dotdict

        data = _make_synthetic_data(n_train=32, n_classes=5, n_classes_h=5,
                                    with_unlabeled=False)
        config = Dotdict({
            "training": {
                "lr": 1e-4, "head_lr": 1e-4, "max_lr": 3e-4,
                "apply_scheduler": False, "epochs": 1, "min_epochs": 1,
                "patience": 5, "lr_schedule_patience": 5,
                "accumulate_grad_batches": 1, "deterministic": True,
                "log_every_n_steps": 1,
            },
            "dataset": {"name": "test", "folder_name": "test", "num_classes": 5},
        })

        transforms = Preprocess()
        dm = BaseDataModule(data, batch_size=4, num_workers=0, transforms=transforms)
        backbone = h_deit_base_embedding(num_classes=5, pretrained=False)
        model = SupervisedModel(config, backbone, num_leaves=5, learning_task="mlc")

        trainer = L.Trainer(
            accelerator="gpu",
            devices=[3, 6],
            strategy="ddp_find_unused_parameters_true",
            max_epochs=1,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            sync_batchnorm=True,
        )
        trainer.fit(model, datamodule=dm)
        assert trainer.current_epoch == 1

    def test_full_helm_ddp(self):
        import lightning as L
        from trainers.ssl_graph_byol import SemiSupervisedGraphBYOLModel
        from models.model import h_deit_base_embedding
        from augmentations import Preprocess
        from datamodules.base_datamodule import BaseDataModule
        from utils.utils import Dotdict

        data = _make_synthetic_data(n_train=32, n_classes=5, n_classes_h=8,
                                    with_unlabeled=True)
        ei = _make_edge_index()
        config = Dotdict({
            "training": {
                "lr": 1e-4, "head_lr": 1e-4, "max_lr": 3e-4,
                "apply_scheduler": False, "epochs": 1, "min_epochs": 1,
                "patience": 5, "lr_schedule_patience": 5,
                "accumulate_grad_batches": 1, "deterministic": True,
                "log_every_n_steps": 1,
            },
            "dataset": {"name": "test", "folder_name": "test", "num_classes": 5},
        })

        transforms = Preprocess()
        dm = BaseDataModule(data, batch_size=4, num_workers=0, transforms=transforms)
        backbone = h_deit_base_embedding(num_classes=8, pretrained=False)
        model = SemiSupervisedGraphBYOLModel(
            config, backbone, num_leaves=5,
            learning_task="hmlc", edge_index=ei,
        )

        trainer = L.Trainer(
            accelerator="gpu",
            devices=[3, 6],
            strategy="ddp_find_unused_parameters_true",
            max_epochs=1,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            sync_batchnorm=True,
        )
        trainer.fit(model, datamodule=dm)
        assert trainer.current_epoch == 1
