"""
Tests for utils/utils.py — Dotdict, metrics calculation, predict helper.
"""
import numpy as np
import pandas as pd
import pytest

from utils.utils import Dotdict, calculate_metrics, OneError, findmax


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Dotdict
# ═══════════════════════════════════════════════════════════════════════════

class TestDotdict:
    def test_simple_access(self):
        d = Dotdict({"a": 1, "b": "hello"})
        assert d.a == 1
        assert d.b == "hello"

    def test_nested_access(self):
        d = Dotdict({"outer": {"inner": 42}})
        assert d.outer.inner == 42

    def test_to_dict_roundtrip(self):
        original = {"a": 1, "b": {"c": 2, "d": {"e": 3}}}
        d = Dotdict(original)
        assert d.to_dict() == original

    def test_missing_key_raises(self):
        d = Dotdict({"a": 1})
        with pytest.raises(AttributeError):
            _ = d.nonexistent

    def test_empty_dict(self):
        d = Dotdict({})
        assert d.to_dict() == {}


# ═══════════════════════════════════════════════════════════════════════════
# 2.  findmax
# ═══════════════════════════════════════════════════════════════════════════

class TestFindmax:
    def test_basic(self):
        arr = np.array([1.0, 3.0, 2.0])
        val, idx = findmax(arr)
        assert val == 3.0
        assert idx == 1

    def test_single_element(self):
        arr = np.array([5.0])
        val, idx = findmax(arr)
        assert val == 5.0
        assert idx == 0

    def test_negative_values(self):
        arr = np.array([-3.0, -1.0, -2.0])
        val, idx = findmax(arr)
        assert val == -1.0
        assert idx == 1


# ═══════════════════════════════════════════════════════════════════════════
# 3.  OneError
# ═══════════════════════════════════════════════════════════════════════════

class TestOneError:
    def test_perfect_predictions(self):
        """When top prediction is always correct, OneError should be 0."""
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        y_scores = np.array([[0.9, 0.1, 0.0], [0.1, 0.8, 0.1]])
        assert OneError(y_scores, y_true) == 0.0

    def test_worst_predictions(self):
        """When top prediction is always wrong, OneError should be 1."""
        y_true = np.array([[1, 0, 0], [0, 1, 0]])
        y_scores = np.array([[0.1, 0.9, 0.0], [0.9, 0.1, 0.1]])
        assert OneError(y_scores, y_true) == 1.0

    def test_skips_all_zeros_and_all_ones(self):
        """Samples where all or no labels are active should be skipped."""
        y_true = np.array([[0, 0, 0], [1, 1, 1], [1, 0, 0]])
        y_scores = np.array([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2], [0.9, 0.1, 0.0]])
        # Only sample 2 (index 2) counts; top-1 correct → 0.0
        assert OneError(y_scores, y_true) == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 4.  calculate_metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestCalculateMetrics:
    def _make_Y(self, n_samples=50, n_classes=10):
        np.random.seed(42)
        y_true = np.zeros((n_samples, n_classes), dtype=int)
        for i in range(n_samples):
            active = np.random.choice(n_classes, size=np.random.randint(1, 4), replace=False)
            y_true[i, active] = 1
        y_scores = np.random.rand(n_samples, n_classes).astype(np.float32)
        y_pred = (y_scores > 0.5).astype(int)
        return {"y_true": y_true, "y_pred": y_pred, "y_scores": y_scores}

    def test_returns_dataframe(self):
        Y = self._make_Y()
        df = calculate_metrics(Y)
        assert isinstance(df, pd.DataFrame)

    def test_all_expected_metrics_present(self):
        Y = self._make_Y()
        df = calculate_metrics(Y)
        expected = [
            "ranking loss", "one error", "coverage",
            "average auprc", "weighted auprc",
            "micro f1", "micro recall", "micro precision",
            "macro f1", "macro recall", "macro precision",
            "subset accuracy", "hamming loss",
            "ml_f_one", "ml_recall", "ml_precision",
        ]
        for name in expected:
            assert name in df.index, f"Missing metric: {name}"

    def test_metrics_values_bounded(self):
        Y = self._make_Y()
        df = calculate_metrics(Y)
        # Most metrics should be in [0, 1] range
        for metric in ["micro f1", "macro f1", "hamming loss", "subset accuracy"]:
            val = df.loc[metric].values[0]
            assert 0.0 <= val <= 1.0, f"{metric}={val} out of [0,1]"

    def test_perfect_predictions(self):
        """When predictions exactly match truth, F1 should be 1.0."""
        n = 20
        y = np.eye(5, dtype=int)
        y = np.tile(y, (4, 1))  # 20 samples
        Y = {"y_true": y, "y_pred": y, "y_scores": y.astype(float)}
        df = calculate_metrics(Y)
        assert df.loc["micro f1"].values[0] == pytest.approx(1.0)
        assert df.loc["subset accuracy"].values[0] == pytest.approx(1.0)
