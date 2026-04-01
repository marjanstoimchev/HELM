"""
Shared fixtures and skip markers for HELM tests.

Dependencies are checked once; tests that need missing packages are
automatically skipped with a clear message.
"""
import os
import sys
import importlib
import pytest
import yaml
import numpy as np

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so ``from data.hierarchy import …``
# style imports work regardless of where pytest is invoked.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Dependency probes — one per optional package
# ---------------------------------------------------------------------------
def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except ImportError:
        return False

HAS_TORCH = _importable("torch")
HAS_LIGHTNING = _importable("lightning")
HAS_TIMM = _importable("timm")
HAS_KORNIA = _importable("kornia")
HAS_PYG = _importable("torch_geometric")
HAS_LIGHTLY = _importable("lightly")
HAS_OMEGACONF = _importable("omegaconf")
HAS_ITERSTRAT = _importable("iterstrat")
HAS_ALBUMENTATIONS = _importable("albumentations")

HAS_CUDA = False
if HAS_TORCH:
    import torch
    HAS_CUDA = torch.cuda.is_available()

HAS_MULTI_GPU = HAS_CUDA and torch.cuda.device_count() > 1 if HAS_TORCH else False

# Composite markers
HAS_FULL_STACK = all([
    HAS_TORCH, HAS_LIGHTNING, HAS_TIMM, HAS_KORNIA,
    HAS_PYG, HAS_LIGHTLY, HAS_OMEGACONF,
])

# Skip decorators
requires_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
requires_lightning = pytest.mark.skipif(not HAS_LIGHTNING, reason="lightning not installed")
requires_timm = pytest.mark.skipif(not HAS_TIMM, reason="timm not installed")
requires_kornia = pytest.mark.skipif(not HAS_KORNIA, reason="kornia not installed")
requires_pyg = pytest.mark.skipif(not HAS_PYG, reason="torch_geometric not installed")
requires_lightly = pytest.mark.skipif(not HAS_LIGHTLY, reason="lightly not installed")
requires_omegaconf = pytest.mark.skipif(not HAS_OMEGACONF, reason="omegaconf not installed")
requires_iterstrat = pytest.mark.skipif(not HAS_ITERSTRAT, reason="iterstrat not installed")
requires_albumentations = pytest.mark.skipif(not HAS_ALBUMENTATIONS, reason="albumentations not installed")
requires_full_stack = pytest.mark.skipif(not HAS_FULL_STACK, reason="full ML stack not installed")
requires_cuda = pytest.mark.skipif(not HAS_CUDA, reason="CUDA not available")
requires_multi_gpu = pytest.mark.skipif(not HAS_MULTI_GPU, reason="multiple GPUs not available")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CONFIGS_DIR = os.path.join(PROJECT_ROOT, "configs")
DATASET_CONFIGS_DIR = os.path.join(CONFIGS_DIR, "dataset")
METHOD_CONFIGS_DIR = os.path.join(CONFIGS_DIR, "method")

ALL_DATASET_CONFIGS = sorted([
    f for f in os.listdir(DATASET_CONFIGS_DIR)
    if f.endswith(".yaml") and not f.startswith(".") and not f.endswith("_.yaml")
]) if os.path.isdir(DATASET_CONFIGS_DIR) else []

ALL_METHOD_CONFIGS = sorted([
    f for f in os.listdir(METHOD_CONFIGS_DIR)
    if f.endswith(".yaml") and not f.startswith(".")
]) if os.path.isdir(METHOD_CONFIGS_DIR) else []

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def simple_hierarchy():
    """A small hierarchy useful for unit tests."""
    return {
        "Root_A": {
            "Mid_A1": {
                "leaf_a": {},
                "leaf_b": {},
            },
            "Mid_A2": {
                "leaf_c": {},
            },
        },
        "Root_B": {
            "leaf_d": {},
            "leaf_e": {},
        },
    }


@pytest.fixture
def ucm_config():
    """Load the UCM dataset config."""
    path = os.path.join(DATASET_CONFIGS_DIR, "ucm.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def make_dummy_data():
    """Factory fixture: returns synthetic images/labels for a given shape."""
    def _make(n_samples=50, n_classes=17, img_h=224, img_w=224, channels=3):
        images = np.random.rand(n_samples, channels, img_h, img_w).astype(np.float32)
        # Each sample has 1-3 active labels
        labels = np.zeros((n_samples, n_classes), dtype=np.float32)
        for i in range(n_samples):
            active = np.random.choice(n_classes, size=np.random.randint(1, 4), replace=False)
            labels[i, active] = 1.0
        return images, labels
    return _make


@pytest.fixture
def dummy_config():
    """Minimal config dict matching what trainers expect."""
    return {
        "training": {
            "lr": 1e-4,
            "head_lr": 1e-4,
            "max_lr": 3e-4,
            "apply_scheduler": False,  # disable for unit tests
            "epochs": 2,
            "min_epochs": 1,
            "patience": 5,
            "lr_schedule_patience": 5,
            "accumulate_grad_batches": 1,
            "deterministic": True,
            "log_every_n_steps": 1,
        },
        "dataset": {
            "name": "test_dataset",
            "folder_name": "test",
            "num_classes": 5,
        },
    }
