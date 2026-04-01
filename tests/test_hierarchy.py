"""
Tests for data/hierarchy.py — hierarchy parsing, path extraction, label extension,
edge index creation.  These tests use only stdlib + numpy + yaml (no ML deps).
"""
import os
import yaml
import numpy as np
import pytest

from tests.conftest import DATASET_CONFIGS_DIR, ALL_DATASET_CONFIGS

# ── imports under test ──────────────────────────────────────────────────────
from data.hierarchy import (
    count_all_nodes,
    extract_paths,
    build_hierarchy_mapping,
    extend_ys,
    process_hierarchy_config,
    encode_classes_and_ancestors,
    create_edge_index,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  count_all_nodes
# ═══════════════════════════════════════════════════════════════════════════

class TestCountAllNodes:
    def test_empty_hierarchy(self):
        assert count_all_nodes({}) == 0

    def test_flat_hierarchy(self):
        h = {"a": {}, "b": {}, "c": {}}
        assert count_all_nodes(h) == 3

    def test_nested_hierarchy(self, simple_hierarchy):
        # Root_A(1) + Mid_A1(1) + leaf_a(1) + leaf_b(1) + Mid_A2(1) + leaf_c(1)
        # + Root_B(1) + leaf_d(1) + leaf_e(1) = 9
        assert count_all_nodes(simple_hierarchy) == 9

    def test_single_deep_chain(self):
        h = {"l1": {"l2": {"l3": {"l4": {}}}}}
        assert count_all_nodes(h) == 4


# ═══════════════════════════════════════════════════════════════════════════
# 2.  extract_paths
# ═══════════════════════════════════════════════════════════════════════════

class TestExtractPaths:
    def test_flat_leaves(self):
        h = {"leaf_a": {}, "leaf_b": {}}
        lp, ip = extract_paths(h)
        assert set(lp.keys()) == {"leaf_a", "leaf_b"}
        assert ip == {}  # no intermediates

    def test_simple_hierarchy(self, simple_hierarchy):
        lp, ip = extract_paths(simple_hierarchy)
        # Leaves should be exactly {leaf_a, leaf_b, leaf_c, leaf_d, leaf_e}
        assert set(lp.keys()) == {"leaf_a", "leaf_b", "leaf_c", "leaf_d", "leaf_e"}
        # Intermediates: Root_A, Mid_A1, Mid_A2, Root_B
        assert "Root_A" in ip
        assert "Root_B" in ip
        assert "Mid_A1" in ip
        assert "Mid_A2" in ip

    def test_leaf_paths_are_unique(self, simple_hierarchy):
        lp, ip = extract_paths(simple_hierarchy)
        codes = list(lp.values()) + list(ip.values())
        assert len(codes) == len(set(codes)), "Path codes must be unique"

    def test_leaf_path_starts_with_ancestor_path(self, simple_hierarchy):
        """Each leaf's code should start with its ancestor's code."""
        lp, ip = extract_paths(simple_hierarchy)
        # leaf_a is under Mid_A1 which is under Root_A
        assert lp["leaf_a"].startswith(ip["Mid_A1"])
        assert ip["Mid_A1"].startswith(ip["Root_A"])

    @pytest.mark.parametrize("config_file", ALL_DATASET_CONFIGS)
    def test_all_dataset_configs_extract(self, config_file):
        """Smoke test: extract_paths should succeed on every dataset config."""
        path = os.path.join(DATASET_CONFIGS_DIR, config_file)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        lp, ip = extract_paths(cfg["hierarchy"])
        # All declared leaf_labels must appear in leaf_paths
        for label in cfg["leaf_labels"]:
            assert label in lp, f"{label} missing from leaf_paths in {config_file}"

    @pytest.mark.parametrize("config_file", ALL_DATASET_CONFIGS)
    def test_num_classes_matches_leaf_count(self, config_file):
        """num_classes in config should equal len(leaf_labels)."""
        path = os.path.join(DATASET_CONFIGS_DIR, config_file)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["num_classes"] == len(cfg["leaf_labels"]), (
            f"{config_file}: num_classes={cfg['num_classes']} but "
            f"len(leaf_labels)={len(cfg['leaf_labels'])}"
        )

    @pytest.mark.parametrize("config_file", ALL_DATASET_CONFIGS)
    def test_leaf_labels_subset_of_hierarchy_leaves(self, config_file):
        """Every leaf_label should appear as a leaf in the hierarchy."""
        path = os.path.join(DATASET_CONFIGS_DIR, config_file)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        lp, _ = extract_paths(cfg["hierarchy"])
        for label in cfg["leaf_labels"]:
            assert label in lp, (
                f"{config_file}: leaf_label '{label}' is not a leaf in the hierarchy"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 3.  build_hierarchy_mapping
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildHierarchyMapping:
    def test_each_label_maps_to_itself(self):
        strings = {"a": "1", "b": "2"}
        mapping = build_hierarchy_mapping(strings)
        for label, path in strings.items():
            assert path in mapping[label]

    def test_ancestors_are_prefixes(self):
        strings = {"leaf": "112", "mid": "11", "root": "1"}
        mapping = build_hierarchy_mapping(strings)
        # leaf should have ancestors 1, 11, 112
        assert "1" in mapping["leaf"]
        assert "11" in mapping["leaf"]
        assert "112" in mapping["leaf"]


# ═══════════════════════════════════════════════════════════════════════════
# 4.  extend_ys
# ═══════════════════════════════════════════════════════════════════════════

class TestExtendYs:
    def test_leaf_columns_preserved(self, simple_hierarchy):
        lp, ip = extract_paths(simple_hierarchy)
        n_leaves = len(lp)
        ys = np.zeros((3, n_leaves), dtype=np.float32)
        ys[0, 0] = 1.0  # first leaf active for sample 0

        ordered_lp = {k: lp[k] for k in sorted(lp.keys())}
        final = {**ordered_lp, **ip}

        ys_ext = extend_ys(ys, ordered_lp, final)
        # First n_leaves columns should match original
        np.testing.assert_array_equal(ys_ext[:, :n_leaves], ys)

    def test_ancestors_activated(self, simple_hierarchy):
        lp, ip = extract_paths(simple_hierarchy)
        leaf_list = sorted(lp.keys())
        ordered_lp = {k: lp[k] for k in leaf_list}
        final = {**ordered_lp, **ip}
        n_leaves = len(ordered_lp)

        ys = np.zeros((1, n_leaves), dtype=np.float32)
        # Activate leaf_a
        leaf_a_idx = leaf_list.index("leaf_a")
        ys[0, leaf_a_idx] = 1.0

        ys_ext = extend_ys(ys, ordered_lp, final)

        # Build reverse: which indices in final correspond to ancestors of leaf_a
        path_to_idx = {path: i for i, path in enumerate(final.values())}
        leaf_a_code = ordered_lp["leaf_a"]
        # All prefixes of leaf_a_code should be active
        for length in range(1, len(leaf_a_code) + 1):
            prefix = leaf_a_code[:length]
            if prefix in path_to_idx:
                assert ys_ext[0, path_to_idx[prefix]] == 1.0, (
                    f"Ancestor with code '{prefix}' should be active"
                )

    def test_inactive_leaf_no_ancestor_activation(self, simple_hierarchy):
        lp, ip = extract_paths(simple_hierarchy)
        leaf_list = sorted(lp.keys())
        ordered_lp = {k: lp[k] for k in leaf_list}
        final = {**ordered_lp, **ip}
        n_leaves = len(ordered_lp)

        # All zeros
        ys = np.zeros((1, n_leaves), dtype=np.float32)
        ys_ext = extend_ys(ys, ordered_lp, final)

        # Everything should remain zero
        np.testing.assert_array_equal(ys_ext, np.zeros_like(ys_ext))

    def test_extended_shape(self, simple_hierarchy):
        lp, ip = extract_paths(simple_hierarchy)
        ordered_lp = {k: lp[k] for k in sorted(lp.keys())}
        final = {**ordered_lp, **ip}
        n_leaves = len(ordered_lp)
        n_total = len(final)

        ys = np.zeros((5, n_leaves), dtype=np.float32)
        ys_ext = extend_ys(ys, ordered_lp, final)
        assert ys_ext.shape == (5, n_total)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  encode_classes_and_ancestors
# ═══════════════════════════════════════════════════════════════════════════

class TestEncodeClassesAndAncestors:
    def test_output_keys_are_leaf_indices(self, simple_hierarchy):
        lp, ip = extract_paths(simple_hierarchy)
        result = encode_classes_and_ancestors(lp, ip)
        assert set(result.keys()) == set(range(len(lp)))

    def test_ancestor_indices_start_after_leaves(self, simple_hierarchy):
        lp, ip = extract_paths(simple_hierarchy)
        n_leaves = len(lp)
        result = encode_classes_and_ancestors(lp, ip)
        for leaf_idx, ancestors in result.items():
            for a in ancestors:
                assert a >= n_leaves, f"Ancestor index {a} should be >= {n_leaves}"

    @pytest.mark.parametrize("config_file", ALL_DATASET_CONFIGS)
    def test_encoding_for_all_datasets(self, config_file):
        path = os.path.join(DATASET_CONFIGS_DIR, config_file)
        with open(path) as f:
            cfg = yaml.safe_load(f)
        lp, ip = extract_paths(cfg["hierarchy"])
        result = encode_classes_and_ancestors(lp, ip)
        assert len(result) == len(lp)


# ═══════════════════════════════════════════════════════════════════════════
# 6.  process_hierarchy_config  (end‑to‑end)
# ═══════════════════════════════════════════════════════════════════════════

class TestProcessHierarchyConfig:
    @pytest.mark.parametrize("config_file", ALL_DATASET_CONFIGS)
    def test_all_configs_parse(self, config_file):
        path = os.path.join(DATASET_CONFIGS_DIR, config_file)
        output = process_hierarchy_config(path)
        assert output.num_classes > 0
        assert output.num_classes_extended >= output.num_classes
        assert len(output.leaf_labels) == output.num_classes

    @pytest.mark.parametrize("config_file", ALL_DATASET_CONFIGS)
    def test_extended_classes_gt_leaf_classes(self, config_file):
        """Extended count should be strictly greater for non-flat hierarchies."""
        path = os.path.join(DATASET_CONFIGS_DIR, config_file)
        output = process_hierarchy_config(path)
        # Every dataset in this repo has intermediate nodes
        assert output.num_classes_extended > output.num_classes, (
            f"{config_file}: expected num_classes_extended > num_classes"
        )

    def test_ucm_specific_values(self):
        path = os.path.join(DATASET_CONFIGS_DIR, "ucm.yaml")
        output = process_hierarchy_config(path)
        assert output.num_classes == 17
        assert output.dataset_name == "UC_Merced_LandUse_Multilabel"


# ═══════════════════════════════════════════════════════════════════════════
# 7.  create_edge_index
# ═══════════════════════════════════════════════════════════════════════════

class TestCreateEdgeIndex:
    def test_hierarchy_edge_index_shape(self, simple_hierarchy):
        lp, ip = extract_paths(simple_hierarchy)
        label_to_pred = encode_classes_and_ancestors(lp, ip)
        ei = create_edge_index(hierarchy=label_to_pred)
        assert ei.shape[0] == 2, "edge_index should have 2 rows"
        assert ei.shape[1] > 0, "should have at least one edge"

    def test_raises_when_no_input(self):
        with pytest.raises(ValueError):
            create_edge_index(hierarchy=None, labeled_data=None)

    @pytest.mark.parametrize("config_file", ALL_DATASET_CONFIGS)
    def test_edge_index_for_all_datasets(self, config_file):
        path = os.path.join(DATASET_CONFIGS_DIR, config_file)
        output = process_hierarchy_config(path)
        label_to_pred = output.label_to_predecessors.to_dict()
        ei = create_edge_index(hierarchy=label_to_pred)
        assert ei.shape[0] == 2
        assert ei.dtype.is_floating_point is False  # should be long

    def test_edge_index_values_in_range(self, simple_hierarchy):
        lp, ip = extract_paths(simple_hierarchy)
        label_to_pred = encode_classes_and_ancestors(lp, ip)
        ei = create_edge_index(hierarchy=label_to_pred)
        total_nodes = len(lp) + len(ip)
        assert ei.max().item() < total_nodes
        assert ei.min().item() >= 0
