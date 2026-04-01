"""
Tests for models/losses.py — all loss functions.
"""
import torch
import pytest

from models.losses import (
    ZLPRLoss,
    CosineSimilarity,
    AsymmetricLoss,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  CosineSimilarity (Negative Cosine Similarity)
# ═══════════════════════════════════════════════════════════════════════════

class TestCosineSimilarity:
    def test_identical_vectors_zero_loss(self):
        loss_fn = CosineSimilarity()
        x = torch.randn(8, 128)
        loss = loss_fn(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_opposite_vectors_max_loss(self):
        loss_fn = CosineSimilarity()
        x = torch.randn(8, 128)
        loss = loss_fn(x, -x)
        assert loss.item() == pytest.approx(4.0, abs=1e-5)

    def test_output_is_scalar(self):
        loss_fn = CosineSimilarity()
        x0 = torch.randn(4, 64)
        x1 = torch.randn(4, 64)
        loss = loss_fn(x0, x1)
        assert loss.dim() == 0

    def test_differentiable(self):
        loss_fn = CosineSimilarity()
        x0 = torch.randn(4, 64, requires_grad=True)
        x1 = torch.randn(4, 64, requires_grad=True)
        loss = loss_fn(x0, x1)
        loss.backward()
        assert x0.grad is not None
        assert x1.grad is not None


# ═══════════════════════════════════════════════════════════════════════════
# 2.  AsymmetricLoss
# ═══════════════════════════════════════════════════════════════════════════

class TestAsymmetricLoss:
    def test_output_is_scalar(self):
        loss_fn = AsymmetricLoss()
        x = torch.randn(8, 10)
        y = torch.zeros(8, 10)
        y[:, :3] = 1.0
        loss = loss_fn(x, y)
        assert loss.dim() == 0

    def test_loss_positive(self):
        loss_fn = AsymmetricLoss()
        x = torch.randn(8, 10)
        y = torch.zeros(8, 10)
        y[:, :3] = 1.0
        loss = loss_fn(x, y)
        assert loss.item() > 0

    def test_differentiable(self):
        loss_fn = AsymmetricLoss()
        x = torch.randn(8, 10, requires_grad=True)
        y = torch.zeros(8, 10)
        y[:, :3] = 1.0
        loss = loss_fn(x, y)
        loss.backward()
        assert x.grad is not None

    def test_perfect_predictions_low_loss(self):
        loss_fn = AsymmetricLoss()
        y = torch.zeros(8, 10)
        y[:, :3] = 1.0
        # High logits where labels are 1, low where 0
        x = torch.full((8, 10), -10.0)
        x[:, :3] = 10.0
        loss = loss_fn(x, y)
        # Loss should be very small for perfect predictions
        assert loss.item() < 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 3.  ZLPRLoss
# ═══════════════════════════════════════════════════════════════════════════

class TestZLPRLoss:
    def test_output_is_scalar(self):
        loss_fn = ZLPRLoss()
        y_pred = torch.randn(8, 10)
        y_true = torch.zeros(8, 10)
        y_true[:, :3] = 1.0
        loss = loss_fn(y_pred, y_true)
        assert loss.dim() == 0

    def test_differentiable(self):
        loss_fn = ZLPRLoss()
        y_pred = torch.randn(8, 10, requires_grad=True)
        y_true = torch.zeros(8, 10)
        y_true[:, :3] = 1.0
        loss = loss_fn(y_pred, y_true)
        loss.backward()
        assert y_pred.grad is not None


# ═══════════════════════════════════════════════════════════════════════════
# 4.  BCEWithLogitsLoss (standard torch, used by all trainers)
# ═══════════════════════════════════════════════════════════════════════════

class TestBCEWithLogitsLoss:
    def test_basic(self):
        loss_fn = torch.nn.BCEWithLogitsLoss()
        x = torch.randn(8, 10)
        y = torch.zeros(8, 10)
        y[:, :3] = 1.0
        loss = loss_fn(x, y)
        assert loss.dim() == 0
        assert loss.item() > 0
