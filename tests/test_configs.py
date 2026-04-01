"""
Tests for configuration consistency — validate all YAML configs,
cross-check method/dataset/training configs, and verify the trainer registry.
"""
import os
import yaml
import pytest

from tests.conftest import (
    CONFIGS_DIR, DATASET_CONFIGS_DIR, METHOD_CONFIGS_DIR,
    ALL_DATASET_CONFIGS, ALL_METHOD_CONFIGS,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Dataset config structure
# ═══════════════════════════════════════════════════════════════════════════

class TestDatasetConfigs:
    REQUIRED_KEYS = {"name", "folder_name", "num_classes", "hierarchy", "leaf_labels"}

    @pytest.mark.parametrize("config_file", ALL_DATASET_CONFIGS)
    def test_required_keys_present(self, config_file):
        path = os.path.join(DATASET_CONFIGS_DIR, config_file)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        missing = self.REQUIRED_KEYS - set(cfg.keys())
        assert not missing, f"{config_file}: missing keys {missing}"

    @pytest.mark.parametrize("config_file", ALL_DATASET_CONFIGS)
    def test_num_classes_positive(self, config_file):
        path = os.path.join(DATASET_CONFIGS_DIR, config_file)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["num_classes"] > 0

    @pytest.mark.parametrize("config_file", ALL_DATASET_CONFIGS)
    def test_leaf_labels_is_list(self, config_file):
        path = os.path.join(DATASET_CONFIGS_DIR, config_file)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg["leaf_labels"], list)

    @pytest.mark.parametrize("config_file", ALL_DATASET_CONFIGS)
    def test_no_duplicate_leaf_labels(self, config_file):
        path = os.path.join(DATASET_CONFIGS_DIR, config_file)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        labels = cfg["leaf_labels"]
        assert len(labels) == len(set(labels)), f"Duplicate leaf labels in {config_file}"

    @pytest.mark.parametrize("config_file", ALL_DATASET_CONFIGS)
    def test_hierarchy_is_dict(self, config_file):
        path = os.path.join(DATASET_CONFIGS_DIR, config_file)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg["hierarchy"], dict)

    @pytest.mark.parametrize("config_file", ALL_DATASET_CONFIGS)
    def test_folder_name_alphanumeric(self, config_file):
        """folder_name should be simple (alphanumeric + underscores)."""
        path = os.path.join(DATASET_CONFIGS_DIR, config_file)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        fn = cfg["folder_name"]
        assert fn.replace("_", "").replace("-", "").isalnum(), (
            f"{config_file}: folder_name '{fn}' contains special characters"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Method config structure
# ═══════════════════════════════════════════════════════════════════════════

class TestMethodConfigs:
    REQUIRED_KEYS = {"name", "learning_task", "training_mode", "apply_edges", "apply_byol"}
    VALID_LEARNING_TASKS = {"mlc", "hmlc"}
    VALID_TRAINING_MODES = {"sl", "ssl"}

    @pytest.mark.parametrize("config_file", ALL_METHOD_CONFIGS)
    def test_required_keys_present(self, config_file):
        path = os.path.join(METHOD_CONFIGS_DIR, config_file)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        missing = self.REQUIRED_KEYS - set(cfg.keys())
        assert not missing, f"{config_file}: missing keys {missing}"

    @pytest.mark.parametrize("config_file", ALL_METHOD_CONFIGS)
    def test_valid_learning_task(self, config_file):
        path = os.path.join(METHOD_CONFIGS_DIR, config_file)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["learning_task"] in self.VALID_LEARNING_TASKS

    @pytest.mark.parametrize("config_file", ALL_METHOD_CONFIGS)
    def test_valid_training_mode(self, config_file):
        path = os.path.join(METHOD_CONFIGS_DIR, config_file)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["training_mode"] in self.VALID_TRAINING_MODES

    @pytest.mark.parametrize("config_file", ALL_METHOD_CONFIGS)
    def test_apply_flags_are_bool(self, config_file):
        path = os.path.join(METHOD_CONFIGS_DIR, config_file)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert isinstance(cfg["apply_edges"], bool)
        assert isinstance(cfg["apply_byol"], bool)

    @pytest.mark.parametrize("config_file", ALL_METHOD_CONFIGS)
    def test_name_matches_filename(self, config_file):
        path = os.path.join(METHOD_CONFIGS_DIR, config_file)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        expected_name = config_file.replace(".yaml", "")
        assert cfg["name"] == expected_name, (
            f"Config name '{cfg['name']}' doesn't match filename '{expected_name}'"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Training config structure
# ═══════════════════════════════════════════════════════════════════════════

class TestTrainingConfig:
    def test_default_training_exists(self):
        path = os.path.join(CONFIGS_DIR, "training", "default.yaml")
        assert os.path.exists(path)

    def test_default_training_keys(self):
        path = os.path.join(CONFIGS_DIR, "training", "default.yaml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        required = {"epochs", "lr", "patience", "batch_size_map"}
        missing = required - set(cfg.keys())
        assert not missing, f"Missing training keys: {missing}"

    def test_batch_size_map_has_standard_fractions(self):
        path = os.path.join(CONFIGS_DIR, "training", "default.yaml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        bsm = cfg["batch_size_map"]
        for frac in [1, 5, 10, 25]:
            assert frac in bsm, f"batch_size_map missing fraction {frac}"

    def test_lr_positive(self):
        path = os.path.join(CONFIGS_DIR, "training", "default.yaml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert float(cfg["lr"]) > 0
        assert float(cfg["max_lr"]) >= float(cfg["lr"])


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Main config.yaml
# ═══════════════════════════════════════════════════════════════════════════

class TestMainConfig:
    def test_main_config_exists(self):
        path = os.path.join(CONFIGS_DIR, "config.yaml")
        assert os.path.exists(path)

    def test_main_config_has_defaults(self):
        path = os.path.join(CONFIGS_DIR, "config.yaml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert "defaults" in cfg

    def test_experiment_section(self):
        path = os.path.join(CONFIGS_DIR, "config.yaml")
        with open(path) as f:
            cfg = yaml.safe_load(f)
        exp = cfg.get("experiment", {})
        assert "seeds" in exp
        assert "fractions" in exp
        assert isinstance(exp["seeds"], list)
        assert isinstance(exp["fractions"], list)
        assert all(0 < f <= 1 for f in exp["fractions"])


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Trainer registry consistency
# ═══════════════════════════════════════════════════════════════════════════

class TestTrainerRegistry:
    """Verify that every method config maps to a registered trainer."""

    @pytest.mark.parametrize("config_file", ALL_METHOD_CONFIGS)
    def test_method_has_registered_trainer(self, config_file):
        from trainers import TRAINER_REGISTRY
        path = os.path.join(METHOD_CONFIGS_DIR, config_file)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        key = (
            cfg["training_mode"],
            cfg["learning_task"],
            cfg["apply_edges"],
            cfg["apply_byol"],
        )
        assert key in TRAINER_REGISTRY, (
            f"No trainer registered for {config_file}: {key}"
        )
