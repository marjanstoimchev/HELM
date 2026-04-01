"""
Tests for callbacks.py — ModelCheckpoint_, EarlyStopping_, RichProgressBar_.
Requires lightning.
"""
import pytest
from tests.conftest import requires_lightning


@requires_lightning
class TestCallbacks:
    def test_model_checkpoint_instantiation(self):
        from callbacks import ModelCheckpoint_
        ckpt = ModelCheckpoint_(dirpath="/tmp/test_ckpt", metric="val_loss", mode="min")
        assert ckpt.monitor == "val_loss"
        assert ckpt.mode == "min"
        assert ckpt.save_top_k == 1

    def test_early_stopping_instantiation(self):
        from callbacks import EarlyStopping_
        es = EarlyStopping_(metric="val_loss", mode="min", patience=5)
        assert es.monitor == "val_loss"
        assert es.patience == 5

    def test_rich_progress_bar_instantiation(self):
        from callbacks import RichProgressBar_
        bar = RichProgressBar_()
        assert bar is not None
