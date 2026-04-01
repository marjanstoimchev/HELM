"""
Tests for data/utils.py — dataset splitting, random sampling, factory.
"""
import numpy as np
import pandas as pd
import pytest

from tests.conftest import requires_iterstrat


# ═══════════════════════════════════════════════════════════════════════════
# 1.  random_sampling
# ═══════════════════════════════════════════════════════════════════════════

class TestRandomSampling:
    def _make_df(self, n=100):
        return pd.DataFrame({"x": range(n)})

    def test_import(self):
        from data.utils import random_sampling
        assert callable(random_sampling)

    def test_fraction_split_sizes(self):
        from data.utils import random_sampling
        df = self._make_df(100)
        labeled, unlabeled = random_sampling(df, p=0.1, seed=42)
        assert len(labeled) == 10
        assert len(unlabeled) == 90

    def test_no_overlap(self):
        from data.utils import random_sampling
        df = self._make_df(100)
        labeled, unlabeled = random_sampling(df, p=0.25, seed=42)
        assert len(set(labeled) & set(unlabeled)) == 0

    def test_all_indices_covered(self):
        from data.utils import random_sampling
        df = self._make_df(100)
        labeled, unlabeled = random_sampling(df, p=0.5, seed=42)
        combined = set(labeled) | set(unlabeled)
        assert combined == set(df.index)

    def test_deterministic(self):
        from data.utils import random_sampling
        df = self._make_df(100)
        l1, u1 = random_sampling(df, p=0.1, seed=42)
        l2, u2 = random_sampling(df, p=0.1, seed=42)
        np.testing.assert_array_equal(sorted(l1), sorted(l2))

    def test_minimum_one_labeled(self):
        from data.utils import random_sampling
        df = self._make_df(10)
        labeled, _ = random_sampling(df, p=0.01, seed=42)
        assert len(labeled) >= 1

    def test_fraction_one(self):
        from data.utils import random_sampling
        df = self._make_df(50)
        labeled, unlabeled = random_sampling(df, p=1.0, seed=42)
        # With p=1.0, n_labeled = floor(50*1.0) = 50 but clamped to 49
        assert len(labeled) + len(unlabeled) == 50


# ═══════════════════════════════════════════════════════════════════════════
# 2.  split_dataset_ (requires iterstrat)
# ═══════════════════════════════════════════════════════════════════════════

@requires_iterstrat
class TestSplitDataset:
    def _make_multilabel_df(self, n=100, n_classes=5):
        np.random.seed(42)
        images = [f"img_{i}.jpg" for i in range(n)]
        one_hots = []
        for _ in range(n):
            oh = np.zeros(n_classes, dtype=int)
            oh[np.random.choice(n_classes, size=np.random.randint(1, 3), replace=False)] = 1
            one_hots.append(oh)
        return pd.DataFrame({"image": images, "one_hot": one_hots})

    def test_split_sizes(self):
        from data.utils import split_dataset_
        df = self._make_multilabel_df(100)
        train_idx, test_idx = split_dataset_(df, test_size=0.2, seed=42)
        assert len(test_idx) == pytest.approx(20, abs=5)
        assert len(train_idx) + len(test_idx) == 100

    def test_no_overlap(self):
        from data.utils import split_dataset_
        df = self._make_multilabel_df(100)
        train_idx, test_idx = split_dataset_(df, test_size=0.2, seed=42)
        assert len(set(train_idx) & set(test_idx)) == 0


# ═══════════════════════════════════════════════════════════════════════════
# 3.  DatasetFactory
# ═══════════════════════════════════════════════════════════════════════════

class TestDatasetFactory:
    def test_unknown_dataset_raises(self):
        from data.utils import DatasetFactory
        with pytest.raises(ValueError, match="Unknown dataset"):
            DatasetFactory.create_dataset("nonexistent_dataset", 10)

    def test_known_names_dont_raise(self):
        """Verify that known names are recognized (instantiation may fail due
        to missing data, but the factory should not raise ValueError)."""
        from data.utils import DatasetFactory
        known = [
            ("dfc_15", 8),
            ("MuRed", 20),
            ("ChestX-ray8", 20),
            ("PadChest", 19),
            ("HPA", 28),
            ("NIHChestXray", 15),
            ("AID_Multilabel", 17),
            ("UC_Merced_LandUse_Multilabel", 17),
            ("MLRSNet", 60),
        ]
        for name, nc in known:
            try:
                obj = DatasetFactory.create_dataset(name, nc)
                assert obj is not None
            except (FileNotFoundError, OSError, Exception) as e:
                # Dataset files may not exist, but factory should still return an object
                # or fail at data loading, not at dispatch
                if "Unknown dataset" in str(e):
                    pytest.fail(f"DatasetFactory doesn't recognize '{name}'")
