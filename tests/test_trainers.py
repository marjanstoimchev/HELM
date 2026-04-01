"""
Tests for all trainer variants — instantiation, forward pass, training/val steps.
Requires the full ML stack (timm, lightning, kornia, torch_geometric, lightly).
"""
import torch
import numpy as np
import pytest

from tests.conftest import (
    requires_full_stack, requires_timm, requires_pyg,
    requires_kornia, requires_lightly, requires_lightning,
)


def _make_config():
    """Minimal config for trainers."""
    from utils.utils import Dotdict
    return Dotdict({
        "training": {
            "lr": 1e-4,
            "head_lr": 1e-4,
            "max_lr": 3e-4,
            "apply_scheduler": False,
            "epochs": 2,
            "min_epochs": 1,
            "patience": 5,
            "lr_schedule_patience": 5,
            "accumulate_grad_batches": 1,
            "deterministic": True,
            "log_every_n_steps": 1,
        },
        "dataset": {
            "name": "test",
            "folder_name": "test",
            "num_classes": 5,
        },
    })


def _make_backbone(num_classes):
    from models.model import h_deit_base_embedding
    return h_deit_base_embedding(num_classes=num_classes, pretrained=False)


def _make_edge_index():
    """Simple edge index for 5 nodes in a chain with 2 ancestors."""
    # 5 leaves + e.g. 2 intermediate = 7 nodes total
    # Edges: 5->0, 5->1, 6->2, 6->3, 6->4
    return torch.tensor([[5, 5, 6, 6, 6],
                         [0, 1, 2, 3, 4]], dtype=torch.long)


def _make_batch(batch_size=2, num_classes=5, num_classes_h=7, with_unlabeled=False):
    """Create a synthetic batch dict."""
    batch = {
        "x": torch.randn(batch_size, 3, 224, 224),
        "one_hot": torch.zeros(batch_size, num_classes).float(),
        "h_one_hot": torch.zeros(batch_size, num_classes_h).float(),
    }
    # Set some labels active
    for i in range(batch_size):
        active = np.random.choice(num_classes, size=2, replace=False)
        batch["one_hot"][i, active] = 1.0
        batch["h_one_hot"][i, active] = 1.0
    if with_unlabeled:
        batch["u"] = torch.randn(batch_size, 3, 224, 224)
    return batch


# ═══════════════════════════════════════════════════════════════════════════
# 1.  SupervisedModel
# ═══════════════════════════════════════════════════════════════════════════

@requires_full_stack
class TestSupervisedModel:
    def test_instantiation_mlc(self):
        from trainers.supervised import SupervisedModel
        cfg = _make_config()
        bb = _make_backbone(5)
        model = SupervisedModel(cfg, bb, num_leaves=5, learning_task="mlc")
        assert model is not None

    def test_instantiation_hmlc(self):
        from trainers.supervised import SupervisedModel
        cfg = _make_config()
        bb = _make_backbone(7)
        model = SupervisedModel(cfg, bb, num_leaves=5, learning_task="hmlc")
        assert model is not None

    def test_forward_mlc(self):
        from trainers.supervised import SupervisedModel
        cfg = _make_config()
        bb = _make_backbone(5)
        model = SupervisedModel(cfg, bb, num_leaves=5, learning_task="mlc")
        model.eval()
        batch = _make_batch(2, 5, 5, with_unlabeled=False)
        with torch.no_grad():
            out = model.forward(batch["x"], batch["one_hot"])
        assert "logits" in out
        assert "loss" in out
        assert out["logits"].shape == (2, 5)
        assert out["loss"].dim() == 0

    def test_forward_hmlc(self):
        from trainers.supervised import SupervisedModel
        cfg = _make_config()
        bb = _make_backbone(7)
        model = SupervisedModel(cfg, bb, num_leaves=5, learning_task="hmlc")
        model.eval()
        batch = _make_batch(2, 5, 7, with_unlabeled=False)
        with torch.no_grad():
            out = model.forward(batch["x"], batch["h_one_hot"])
        assert out["logits"].shape == (2, 5)

    def test_training_step_returns_loss(self):
        from trainers.supervised import SupervisedModel
        cfg = _make_config()
        bb = _make_backbone(5)
        model = SupervisedModel(cfg, bb, num_leaves=5, learning_task="mlc")
        batch = _make_batch(2, 5, 5)
        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_configure_optimizers(self):
        from trainers.supervised import SupervisedModel
        cfg = _make_config()
        bb = _make_backbone(5)
        model = SupervisedModel(cfg, bb, num_leaves=5, learning_task="mlc")
        opt_config = model.configure_optimizers()
        assert "optimizer" in opt_config


# ═══════════════════════════════════════════════════════════════════════════
# 2.  GraphBasedModel
# ═══════════════════════════════════════════════════════════════════════════

@requires_full_stack
class TestGraphBasedModel:
    def test_instantiation(self):
        from trainers.ssl_graph import GraphBasedModel
        cfg = _make_config()
        bb = _make_backbone(7)
        ei = _make_edge_index()
        model = GraphBasedModel(cfg, bb, num_leaves=5, learning_task="hmlc", edge_index=ei)
        assert model is not None

    def test_forward(self):
        from trainers.ssl_graph import GraphBasedModel
        cfg = _make_config()
        bb = _make_backbone(7)
        ei = _make_edge_index()
        model = GraphBasedModel(cfg, bb, num_leaves=5, learning_task="hmlc", edge_index=ei)
        model.eval()
        batch = _make_batch(2, 5, 7)
        with torch.no_grad():
            out = model.forward(batch["x"], batch["h_one_hot"])
        assert "logits" in out
        assert "loss" in out

    def test_training_step(self):
        from trainers.ssl_graph import GraphBasedModel
        cfg = _make_config()
        bb = _make_backbone(7)
        ei = _make_edge_index()
        model = GraphBasedModel(cfg, bb, num_leaves=5, learning_task="hmlc", edge_index=ei)
        batch = _make_batch(2, 5, 7)
        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)


# ═══════════════════════════════════════════════════════════════════════════
# 3.  SemiSupervisedBYOLModel
# ═══════════════════════════════════════════════════════════════════════════

@requires_full_stack
class TestSemiSupervisedBYOLModel:
    def test_instantiation(self):
        from trainers.ssl_byol import SemiSupervisedBYOLModel
        cfg = _make_config()
        bb = _make_backbone(7)
        model = SemiSupervisedBYOLModel(cfg, bb, num_leaves=5)
        assert model is not None
        assert hasattr(model, "projection_head")
        assert hasattr(model, "backbone_momentum")

    def test_forward_with_unlabeled(self):
        from trainers.ssl_byol import SemiSupervisedBYOLModel
        cfg = _make_config()
        bb = _make_backbone(7)
        model = SemiSupervisedBYOLModel(cfg, bb, num_leaves=5)
        model.eval()
        batch = _make_batch(2, 5, 7, with_unlabeled=True)
        with torch.no_grad():
            out = model.forward(batch["x"], batch["h_one_hot"], u=batch["u"])
        assert "logits" in out
        assert "loss" in out
        assert out["logits"].shape[1] == 5

    def test_forward_without_unlabeled(self):
        from trainers.ssl_byol import SemiSupervisedBYOLModel
        cfg = _make_config()
        bb = _make_backbone(7)
        model = SemiSupervisedBYOLModel(cfg, bb, num_leaves=5)
        model.eval()
        batch = _make_batch(2, 5, 7, with_unlabeled=False)
        with torch.no_grad():
            out = model.forward(batch["x"], batch["h_one_hot"])
        assert "logits" in out

    def test_training_step(self):
        from trainers.ssl_byol import SemiSupervisedBYOLModel
        cfg = _make_config()
        bb = _make_backbone(7)
        model = SemiSupervisedBYOLModel(cfg, bb, num_leaves=5)
        batch = _make_batch(2, 5, 7, with_unlabeled=True)
        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  SemiSupervisedGraphBYOLModel (full HELM)
# ═══════════════════════════════════════════════════════════════════════════

@requires_full_stack
class TestSemiSupervisedGraphBYOLModel:
    def test_instantiation(self):
        from trainers.ssl_graph_byol import SemiSupervisedGraphBYOLModel
        cfg = _make_config()
        bb = _make_backbone(7)
        ei = _make_edge_index()
        model = SemiSupervisedGraphBYOLModel(cfg, bb, num_leaves=5, edge_index=ei)
        assert model is not None
        assert hasattr(model, "sage")
        assert hasattr(model, "projection_head")

    def test_forward_with_unlabeled(self):
        from trainers.ssl_graph_byol import SemiSupervisedGraphBYOLModel
        cfg = _make_config()
        bb = _make_backbone(7)
        ei = _make_edge_index()
        model = SemiSupervisedGraphBYOLModel(cfg, bb, num_leaves=5, edge_index=ei)
        model.eval()
        batch = _make_batch(2, 5, 7, with_unlabeled=True)
        with torch.no_grad():
            out = model.forward(batch["x"], batch["h_one_hot"], u=batch["u"])
        assert "logits" in out
        assert "loss" in out
        assert out["logits"].shape[1] == 5

    def test_forward_without_unlabeled(self):
        from trainers.ssl_graph_byol import SemiSupervisedGraphBYOLModel
        cfg = _make_config()
        bb = _make_backbone(7)
        ei = _make_edge_index()
        model = SemiSupervisedGraphBYOLModel(cfg, bb, num_leaves=5, edge_index=ei)
        model.eval()
        batch = _make_batch(2, 5, 7)
        with torch.no_grad():
            out = model.forward(batch["x"], batch["h_one_hot"])
        assert "logits" in out

    def test_training_step(self):
        from trainers.ssl_graph_byol import SemiSupervisedGraphBYOLModel
        cfg = _make_config()
        bb = _make_backbone(7)
        ei = _make_edge_index()
        model = SemiSupervisedGraphBYOLModel(cfg, bb, num_leaves=5, edge_index=ei)
        batch = _make_batch(2, 5, 7, with_unlabeled=True)
        loss = model.training_step(batch, 0)
        assert isinstance(loss, torch.Tensor)

    def test_eval_forward(self):
        from trainers.ssl_graph_byol import SemiSupervisedGraphBYOLModel
        cfg = _make_config()
        bb = _make_backbone(7)
        ei = _make_edge_index()
        model = SemiSupervisedGraphBYOLModel(cfg, bb, num_leaves=5, edge_index=ei)
        model.eval()
        batch = _make_batch(2, 5, 7)
        with torch.no_grad():
            out = model._forward_eval(batch["x"], batch["h_one_hot"])
        assert "logits" in out
        assert "loss" in out


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Trainer factory (get_trainer_class)
# ═══════════════════════════════════════════════════════════════════════════

class TestGetTrainerClass:
    def test_all_valid_combinations(self):
        from trainers import get_trainer_class, TRAINER_REGISTRY
        for key, cls in TRAINER_REGISTRY.items():
            result = get_trainer_class(*key)
            assert result is cls

    def test_invalid_combination_raises(self):
        from trainers import get_trainer_class
        with pytest.raises(ValueError):
            get_trainer_class("sl", "mlc", True, True)  # not registered
