"""
Tests for data/dataset_pipeline.py — DatasetPipeline hierarchy processing.
Dataset loading/splitting are tested separately; here we test the config parsing
path that doesn't require actual dataset files.
"""
import os
import pytest

from tests.conftest import DATASET_CONFIGS_DIR, ALL_DATASET_CONFIGS


class TestDatasetPipelineHierarchy:
    """Test hierarchy processing in DatasetPipeline (doesn't need data files)."""

    @pytest.mark.parametrize("config_file", ALL_DATASET_CONFIGS)
    def test_hierarchy_config_loads(self, config_file):
        from data.dataset_pipeline import DatasetPipeline
        yaml_path = os.path.join(DATASET_CONFIGS_DIR, config_file)
        # DatasetPipeline.__init__ calls process_hierarchy_config_
        pipeline = DatasetPipeline(yaml_path=yaml_path, seed=42)
        assert pipeline.dataset_name is not None
        assert pipeline.num_classes > 0
        assert pipeline.num_classes_extended > 0

    @pytest.mark.parametrize("config_file", ALL_DATASET_CONFIGS)
    def test_leaf_paths_populated(self, config_file):
        from data.dataset_pipeline import DatasetPipeline
        yaml_path = os.path.join(DATASET_CONFIGS_DIR, config_file)
        pipeline = DatasetPipeline(yaml_path=yaml_path, seed=42)
        assert len(pipeline.leaf_paths) > 0

    @pytest.mark.parametrize("config_file", ALL_DATASET_CONFIGS)
    def test_intermediate_paths_populated(self, config_file):
        from data.dataset_pipeline import DatasetPipeline
        yaml_path = os.path.join(DATASET_CONFIGS_DIR, config_file)
        pipeline = DatasetPipeline(yaml_path=yaml_path, seed=42)
        assert len(pipeline.intermediate_paths) > 0

    @pytest.mark.parametrize("config_file", ALL_DATASET_CONFIGS)
    def test_label_to_predecessors_populated(self, config_file):
        from data.dataset_pipeline import DatasetPipeline
        yaml_path = os.path.join(DATASET_CONFIGS_DIR, config_file)
        pipeline = DatasetPipeline(yaml_path=yaml_path, seed=42)
        assert len(pipeline.label_to_predecessors) > 0

    @pytest.mark.parametrize("config_file", ALL_DATASET_CONFIGS)
    def test_num_classes_extended_consistent(self, config_file):
        from data.dataset_pipeline import DatasetPipeline
        yaml_path = os.path.join(DATASET_CONFIGS_DIR, config_file)
        pipeline = DatasetPipeline(yaml_path=yaml_path, seed=42)
        expected = len(pipeline.leaf_paths) + len(pipeline.intermediate_paths)

        # Guard: leaf and intermediate names must not collide, otherwise
        # dict merge silently drops entries (was a bug in chestxray8.yaml).
        overlap = set(pipeline.leaf_paths.keys()) & set(pipeline.intermediate_paths.keys())
        assert not overlap, (
            f"Leaf/intermediate name collision in {config_file}: {overlap}. "
            f"Rename the parent node (e.g. append '_') to avoid dict key overwrite."
        )
        assert pipeline.num_classes_extended == expected, (
            f"num_classes_extended ({pipeline.num_classes_extended}) != "
            f"len(leaf) ({len(pipeline.leaf_paths)}) + len(intermediate) ({len(pipeline.intermediate_paths)})"
        )
