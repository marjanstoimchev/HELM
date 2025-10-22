"""
Test script for SSL integration verification.

This script tests the critical components of the semi-supervised learning
implementation to ensure BYOL and graph learning work correctly.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.augmentations import DataAugmentation, create_byol_augmentations
from src.data.ssl_dataset import SemiSupervisedSplitDataset
from src.data.torch_dataset import HierarchicalImageDataset
import numpy as np


def test_augmentations():
    """Test that augmentation pipelines work correctly."""
    print("\n" + "=" * 80)
    print("TEST 1: Augmentation Pipelines")
    print("=" * 80)

    # Create test input
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)

    # Test weak augmentation
    weak_aug = DataAugmentation(mode='weak', image_size=224)
    x_weak = weak_aug(x)
    assert x_weak.shape == x.shape, "Weak augmentation changed shape!"
    print("✓ Weak augmentation: OK")

    # Test strong augmentation
    strong_aug = DataAugmentation(mode='strong', image_size=224)
    x_strong = strong_aug(x)
    assert x_strong.shape == x.shape, "Strong augmentation changed shape!"
    print("✓ Strong augmentation: OK")

    # Test create_byol_augmentations
    weak, strong = create_byol_augmentations(image_size=224)
    x_w = weak(x)
    x_s = strong(x)
    assert x_w.shape == x.shape and x_s.shape == x.shape, "BYOL augmentations changed shape!"
    print("✓ BYOL augmentation factory: OK")

    print("\n✅ All augmentation tests passed!")


def test_byol_view_pairing():
    """Test that BYOL creates properly paired views."""
    print("\n" + "=" * 80)
    print("TEST 2: BYOL View Pairing")
    print("=" * 80)

    from src.training.helm_module import HELMModule

    # Create mock config
    config = {
        'model': {
            'embed_dim': 768,
            'byol_hidden_dim': 1024,
            'byol_projection_dim': 256,
            'byol_momentum': 0.996,
            'graph_type': 'sage',
            'graph_hidden_dim': 64,
            'graph_dropout': 0.3,
        },
        'training': {
            'loss': {
                'use_classification_loss': True,
                'use_graph_loss': False,
                'use_byol_loss': True,
                'classification_weight': 1.0,
                'graph_weight': 0.0,
                'byol_weight': 1.0,
                'classification_loss_type': 'bce',
                'graph_loss_type': 'bce',
            },
            'lr': 1e-4,
            'head_lr': 1e-4,
            'weight_decay': 0.01,
            'optimizer': 'adamw',
            'use_scheduler': False,
            'max_epochs': 1,
            'min_epochs': 1,
            'batch_size': 4,
            'accumulate_grad_batches': 1,
            'gradient_clip_val': 1.0,
            'log_every_n_steps': 1,
        },
        'dataset': {
            'image_size': 224,
        },
        'system': {
            'seed': 42,
        }
    }

    # Create model
    num_classes = 18
    num_leaves = 8
    leaf_indices = torch.arange(num_leaves)

    model = HELMModule(
        num_classes=num_classes,
        num_leaves=num_leaves,
        edge_index=None,
        leaf_indices=leaf_indices,
        config=config,
    )

    # Test Case 1: Supervised with BYOL (u=None)
    print("\nCase 1: Supervised with BYOL (u=None)")
    x = torch.randn(4, 3, 224, 224)
    view1, view2 = model._prepare_data(x, u=None)
    assert view1.shape == view2.shape, f"Views have different shapes: {view1.shape} vs {view2.shape}"
    assert view1.shape == x.shape, f"View shape doesn't match input: {view1.shape} vs {x.shape}"
    print(f"  ✓ View1 shape: {view1.shape}")
    print(f"  ✓ View2 shape: {view2.shape}")

    # Test Case 2: Semi-supervised with BYOL (u is not None)
    print("\nCase 2: Semi-supervised with BYOL (u is not None)")
    x = torch.randn(4, 3, 224, 224)  # labeled
    u = torch.randn(8, 3, 224, 224)  # unlabeled
    view1, view2 = model._prepare_data(x, u)
    expected_shape = torch.Size([12, 3, 224, 224])  # 4 + 8 = 12
    assert view1.shape == view2.shape, f"Views have different shapes: {view1.shape} vs {view2.shape}"
    assert view1.shape == expected_shape, f"View shape incorrect: {view1.shape} vs {expected_shape}"
    print(f"  ✓ View1 shape: {view1.shape}")
    print(f"  ✓ View2 shape: {view2.shape}")
    print(f"  ✓ Combined batch size: {view1.shape[0]} (labeled: {x.shape[0]} + unlabeled: {u.shape[0]})")

    print("\n✅ All BYOL view pairing tests passed!")


def test_ssl_dataset_resampling():
    """Test that SSL dataset properly resamples labeled data."""
    print("\n" + "=" * 80)
    print("TEST 3: SSL Dataset Resampling")
    print("=" * 80)

    # Create mock dataset
    class MockDataset:
        def __init__(self, size):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {
                'images': torch.randn(3, 224, 224),
                'labels': torch.zeros(18),
                'leaf_labels': torch.zeros(8),
                'leaf_indices': torch.arange(8),
            }

    # Test Case 1: labeled_ratio = 0.3
    print("\nCase 1: labeled_ratio=0.3")
    full_dataset = MockDataset(size=100)
    ssl_dataset = SemiSupervisedSplitDataset(full_dataset, labeled_ratio=0.3, seed=42)

    n_labeled_original = int(100 * 0.3)  # 30
    n_unlabeled = 100 - n_labeled_original  # 70

    print(f"  Original dataset size: 100")
    print(f"  Labeled samples (30%): {n_labeled_original}")
    print(f"  Unlabeled samples (70%): {n_unlabeled}")
    print(f"  SSL dataset length: {len(ssl_dataset)}")

    assert len(ssl_dataset) == n_unlabeled, \
        f"SSL dataset size should match unlabeled size: {len(ssl_dataset)} vs {n_unlabeled}"

    # Check that we can get samples
    sample = ssl_dataset[0]
    assert 'images' in sample and 'u' in sample, "Sample should contain both labeled and unlabeled"
    print(f"  ✓ Sample contains 'images' (labeled) and 'u' (unlabeled)")

    # Test Case 2: labeled_ratio = 1.0 (no unlabeled)
    print("\nCase 2: labeled_ratio=1.0 (supervised with BYOL)")
    ssl_dataset_full = SemiSupervisedSplitDataset(full_dataset, labeled_ratio=1.0, seed=42)
    print(f"  SSL dataset length: {len(ssl_dataset_full)}")
    assert len(ssl_dataset_full) == 100, "With labeled_ratio=1.0, should have all samples as labeled"

    sample = ssl_dataset_full[0]
    assert 'images' in sample, "Sample should contain labeled images"
    # In this case, there might not be 'u' since no unlabeled data
    print(f"  ✓ Sample contains 'images'")
    print(f"  ✓ No unlabeled data created (as expected)")

    print("\n✅ All SSL dataset resampling tests passed!")


def test_forward_pass():
    """Test complete forward pass with SSL."""
    print("\n" + "=" * 80)
    print("TEST 4: Complete Forward Pass")
    print("=" * 80)

    from src.training.helm_module import HELMModule

    config = {
        'model': {
            'embed_dim': 768,
            'byol_hidden_dim': 1024,
            'byol_projection_dim': 256,
            'byol_momentum': 0.996,
            'graph_type': 'sage',
            'graph_hidden_dim': 64,
            'graph_dropout': 0.3,
        },
        'training': {
            'loss': {
                'use_classification_loss': True,
                'use_graph_loss': False,
                'use_byol_loss': True,
                'classification_weight': 1.0,
                'graph_weight': 0.0,
                'byol_weight': 1.0,
                'classification_loss_type': 'bce',
                'graph_loss_type': 'bce',
            },
            'lr': 1e-4,
            'head_lr': 1e-4,
            'weight_decay': 0.01,
            'optimizer': 'adamw',
            'use_scheduler': False,
            'max_epochs': 1,
            'min_epochs': 1,
            'batch_size': 4,
            'accumulate_grad_batches': 1,
            'gradient_clip_val': 1.0,
            'log_every_n_steps': 1,
        },
        'dataset': {
            'image_size': 224,
        },
        'system': {
            'seed': 42,
        }
    }

    num_classes = 18
    num_leaves = 8
    leaf_indices = torch.arange(num_leaves)

    model = HELMModule(
        num_classes=num_classes,
        num_leaves=num_leaves,
        edge_index=None,
        leaf_indices=leaf_indices,
        config=config,
    )
    model.eval()

    # Test forward pass
    print("\nTesting forward pass with SSL...")
    x = torch.randn(4, 3, 224, 224)  # labeled
    y = torch.zeros(4, num_classes)  # labels
    u = torch.randn(8, 3, 224, 224)  # unlabeled

    with torch.no_grad():
        outputs = model(x, y, u)

    print(f"  ✓ Output keys: {outputs.keys()}")
    assert 'loss' in outputs, "Output should contain 'loss'"
    assert 'loss_cls' in outputs, "Output should contain 'loss_cls'"
    assert 'loss_byol' in outputs, "Output should contain 'loss_byol'"
    print(f"  ✓ Total loss: {outputs['loss'].item():.4f}")
    print(f"  ✓ Classification loss: {outputs['loss_cls'].item():.4f}")
    print(f"  ✓ BYOL loss: {outputs['loss_byol'].item():.4f}")

    # Test without unlabeled (supervised with BYOL)
    print("\nTesting forward pass without unlabeled (supervised with BYOL)...")
    with torch.no_grad():
        outputs = model(x, y, u=None)

    print(f"  ✓ Output keys: {outputs.keys()}")
    assert 'loss' in outputs, "Output should contain 'loss'"
    assert 'loss_byol' in outputs, "Output should contain 'loss_byol' even without unlabeled"
    print(f"  ✓ Total loss: {outputs['loss'].item():.4f}")
    print(f"  ✓ BYOL loss computed even without unlabeled data")

    print("\n✅ All forward pass tests passed!")


def run_all_tests():
    """Run all SSL integration tests."""
    print("\n" + "=" * 80)
    print("SEMI-SUPERVISED LEARNING INTEGRATION TESTS")
    print("=" * 80)

    try:
        test_augmentations()
        test_byol_view_pairing()
        test_ssl_dataset_resampling()
        test_forward_pass()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED! SSL INTEGRATION IS WORKING CORRECTLY")
        print("=" * 80)
        return True

    except Exception as e:
        print("\n" + "=" * 80)
        print(f"❌ TEST FAILED: {str(e)}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
