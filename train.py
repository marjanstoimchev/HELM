"""
Main training script for HELM (Hierarchical Embedding Learning Model).

This script orchestrates the complete training pipeline:
1. Loads configuration from YAML files
2. Sets up dataset and hierarchy
3. Initializes model with configurable loss components
4. Trains using PyTorch Lightning

Usage:
    python train.py
    python train.py training=full_ssl  # Use different training config
    python train.py dataset=cub_200_2011  # Use different dataset
"""

import sys
from pathlib import Path
import torch
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import hydra
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import HierarchyManager
from src.data.torch_dataset import HierarchicalImageDataset
from src.data.hierarchies.hierarchy_utils import HierarchyGraphGenerator
from src.data.ssl_dataset import SemiSupervisedSplitDataset
from src.training import HELMModule
from src.utils import setup_experiment


def create_dataloaders(config: DictConfig, use_full_hierarchy: bool = True):
    """Create dataloaders directly without using LightningDataModule.

    Args:
        config: Configuration dict
        use_full_hierarchy: If True, use full hierarchy labels (HMLC mode).
                          If False, use only leaf labels (MLC mode).
    """
    from src.data.dataset.utils import create_dataset
    from src.data.dataset.splitter import stratified_split_dataset
    from src.data.dataset.readers.image_reader import ImageReader
    from torch.utils.data import DataLoader
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import cv2

    dataset_cfg = config['dataset']

    # Load hierarchy
    hierarchy, leaf_labels = HierarchyManager.load_hierarchy_config(
        dataset_cfg['hierarchy_config_dir'],
        dataset_cfg['name']
    )

    # Create dataset
    df = create_dataset(
        dataset_cfg['name'],
        data_dir=dataset_cfg.get('data_dir'),
        leaf_labels=leaf_labels
    )

    # Split dataset
    split_df = stratified_split_dataset(
        df,
        label_cols=leaf_labels,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=config['system']['seed']
    )

    # Prepare dataframes for each split
    dataframes = {
        'train': split_df[split_df['split'] == 'train'].reset_index(drop=True),
        'val': split_df[split_df['split'] == 'val'].reset_index(drop=True),
        'test': split_df[split_df['split'] == 'test'].reset_index(drop=True),
    }

    # Setup image reader
    reader = ImageReader(dataset_cfg['name'])

    # Determine image column
    dataset_name_normalized = dataset_cfg['name'].lower().replace('-', '_')
    huggingface_datasets = {'ucm', 'aid', 'mlrsnet', 'hf', 'stanford_cars'}
    image_column = 'image' if dataset_name_normalized in huggingface_datasets else 'full_path'

    # Create transforms
    image_size = dataset_cfg['image_size']
    augmentation_strength = config['training']['augmentation_strength']

    if augmentation_strength == "default":
        train_transforms = A.Compose([
            A.Affine(scale=(0.85, 1.15), translate_percent=0, rotate=0,
                    interpolation=cv2.INTER_CUBIC, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.SmallestMaxSize(max_size=image_size),
            A.RandomCrop(height=image_size, width=image_size, p=1.0),
            ToTensorV2(),
        ])
    else:
        train_transforms = A.Compose([
            A.Resize(image_size, image_size),
            ToTensorV2()
        ])

    val_test_transforms = A.Compose([
        A.Resize(image_size, image_size),
        ToTensorV2(),
    ])

    # Create datasets
    dataset_kwargs_base = {
        "image_column": image_column,
        "image_reader": reader,
        "leaf_columns": leaf_labels,
        "hierarchy": hierarchy,
        "graph_mode": "upper_triangular",
    }

    train_dataset = HierarchicalImageDataset(
        dataframes['train'],
        transforms=train_transforms,
        **dataset_kwargs_base
    )

    val_dataset = HierarchicalImageDataset(
        dataframes['val'],
        transforms=val_test_transforms,
        **dataset_kwargs_base
    )

    test_dataset = HierarchicalImageDataset(
        dataframes['test'],
        transforms=val_test_transforms,
        **dataset_kwargs_base
    )

    # Check if semi-supervised mode is enabled
    semi_supervised = config['training'].get('semi_supervised', False)
    labeled_ratio = config['training'].get('labeled_ratio', 1.0) if semi_supervised else 1.0

    # Create collate functions (separate for train and val/test)
    def train_collate_fn(batch):
        """
        Training collate function with SSL support.

        For semi-supervised mode, expects batch items with 'u' key containing unlabeled samples.
        For supervised mode, just collates labeled samples.
        """
        import torch
        images = torch.stack([item['images'] for item in batch])

        # Select labels based on mode
        if use_full_hierarchy:
            # HMLC mode: use full hierarchy labels
            labels = torch.stack([item['labels'] for item in batch])
        else:
            # MLC mode: use only leaf labels
            labels = torch.stack([item['leaf_labels'] for item in batch])

        batch_dict = {
            'images': images,
            'labels': labels,
            'leaf_labels': torch.stack([item['leaf_labels'] for item in batch]),
            'leaf_indices': torch.stack([item['leaf_indices'] for item in batch]),
        }

        # For semi-supervised mode, add unlabeled data if present
        if semi_supervised and 'u' in batch[0]:
            unlabeled_images = torch.stack([item['u'] for item in batch])
            batch_dict['u'] = unlabeled_images

        return batch_dict

    def val_collate_fn(batch):
        """
        Validation/test collate function (no unlabeled data).

        Always returns only labeled samples for validation/testing.
        """
        import torch
        images = torch.stack([item['images'] for item in batch])

        # Select labels based on mode
        if use_full_hierarchy:
            labels = torch.stack([item['labels'] for item in batch])
        else:
            labels = torch.stack([item['leaf_labels'] for item in batch])

        batch_dict = {
            'images': images,
            'labels': labels,
            'leaf_labels': torch.stack([item['leaf_labels'] for item in batch]),
            'leaf_indices': torch.stack([item['leaf_indices'] for item in batch]),
        }

        return batch_dict

    # Wrap train dataset with SSL dataset if semi-supervised mode is enabled
    if semi_supervised:
        print(f"Creating semi-supervised dataset with labeled_ratio={labeled_ratio}")
        train_dataset = SemiSupervisedSplitDataset(
            train_dataset,
            labeled_ratio=labeled_ratio,
            seed=config['system']['seed'],
        )

    # Create dataloaders
    batch_size = config['training']['batch_size']
    num_workers = dataset_cfg['num_workers']

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        collate_fn=train_collate_fn  # Use train collate function
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        collate_fn=val_collate_fn  # Use validation collate function
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
        collate_fn=val_collate_fn  # Use validation collate function
    )

    return train_loader, val_loader, test_loader, hierarchy, leaf_labels


def create_model(config: DictConfig, hierarchy, leaf_labels, num_classes: int, num_leaves: int):
    """Create the HELM model with edge index and leaf indices."""
    use_graph_loss = config['training']['loss']['use_graph_loss']
    use_full_hierarchy = config['training'].get('use_full_hierarchy', True)

    # Create graph structure (needed for both edge_index and leaf_indices)
    from src.data.hierarchies.hierarchy_utils import HierarchyGraphGenerator
    graph_gen = HierarchyGraphGenerator(hierarchy, leaf_labels)
    graph_output = graph_gen.generate(mode="upper_triangular")
    node_mapping = graph_output['node_mapping']

    # Get leaf indices in the node_mapping order
    leaf_indices = torch.tensor([node_mapping[leaf] for leaf in leaf_labels], dtype=torch.long)

    if use_graph_loss and use_full_hierarchy:
        # Only use edge_index for HMLC+Graph mode
        edge_index = graph_output['edge_index']
    else:
        edge_index = None

    # Create model
    model = HELMModule(
        num_classes=num_classes,
        num_leaves=num_leaves,
        edge_index=edge_index,
        leaf_indices=leaf_indices,
        config=OmegaConf.to_container(config, resolve=True),
    )

    return model


def create_callbacks(config: DictConfig, exp_dir: Path):
    """Create PyTorch Lightning callbacks."""
    callbacks = []

    # Checkpoint callback - monitor val_loss
    checkpoint_cfg = config['training']['checkpoint']
    checkpoint_callback = ModelCheckpoint(
        dirpath=exp_dir / 'checkpoints',
        filename='helm-{epoch:02d}-{val_loss:.4f}',
        monitor=checkpoint_cfg['monitor'],
        mode=checkpoint_cfg['mode'],
        save_top_k=checkpoint_cfg['save_top_k'],
        save_last=checkpoint_cfg['save_last'],
        verbose=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback - monitor val_loss
    early_stop_cfg = config['training']['early_stopping']
    if early_stop_cfg['enabled']:
        early_stop_callback = EarlyStopping(
            monitor=early_stop_cfg['monitor'],
            patience=early_stop_cfg['patience'],
            mode=early_stop_cfg['mode'],
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    return callbacks


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig):
    """Main training function."""
    # Handle deterministic mode - use warn_only for operations without deterministic implementation
    if config['system']['deterministic']:
        import torch
        torch.use_deterministic_algorithms(True, warn_only=True)

    print("=" * 80)
    print("HELM Training")
    print("=" * 80)
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(config))
    print("=" * 80)

    # Setup experiment
    exp_dir = setup_experiment(OmegaConf.to_container(config, resolve=True))
    print(f"\nExperiment directory: {exp_dir}")

    # Determine which labels to use
    use_full_hierarchy = config['training'].get('use_full_hierarchy', True)

    # Create dataloaders
    print("\nCreating dataloaders...")
    semi_supervised = config['training'].get('semi_supervised', False)
    if semi_supervised:
        labeled_ratio = config['training'].get('labeled_ratio', 1.0)
        print(f"Semi-supervised mode: {labeled_ratio*100:.0f}% labeled, {(1-labeled_ratio)*100:.0f}% unlabeled")
    train_loader, val_loader, test_loader, hierarchy, leaf_labels = create_dataloaders(config, use_full_hierarchy)

    # Get dataset info from train dataset
    sample_dataset = train_loader.dataset
    hierarchy_info = sample_dataset.get_hierarchy_info()
    num_classes_total = len(hierarchy_info['node_mapping'])
    num_leaves = len(leaf_labels)

    # Determine num_classes and labels to use based on training mode
    # MLC: use only leaf labels (8 for DFC-15)
    # HMLC_flat: use full hierarchy (18 for DFC-15), no graph
    # HMLC+Graph: use full hierarchy (18 for DFC-15) with graph learning
    use_full_hierarchy = config['training'].get('use_full_hierarchy', True)
    use_graph = config['training']['loss']['use_graph_loss']

    if use_full_hierarchy:
        # HMLC mode (flat or with graph): use full hierarchy
        num_classes = num_classes_total
        mode_name = "HMLC + Graph" if use_graph else "HMLC_flat"
    else:
        # MLC mode: use only leaves
        num_classes = num_leaves
        mode_name = "MLC"

    print(f"\nMode: {mode_name}")
    print(f"Dataset: {config['dataset']['name']}")
    print(f"Total hierarchy nodes: {num_classes_total}")
    print(f"Leaf nodes: {num_leaves}")
    print(f"Classes for classifier: {num_classes}")

    # Create model
    print("\nCreating model...")
    model = create_model(config, hierarchy, leaf_labels, num_classes, num_leaves)

    # Print loss configuration
    loss_cfg = config['training']['loss']
    print("\nLoss Configuration:")
    print(f"  Classification Loss: {'✓' if loss_cfg['use_classification_loss'] else '✗'} "
          f"(weight: {loss_cfg['classification_weight']})")
    print(f"  Graph Loss: {'✓' if loss_cfg['use_graph_loss'] else '✗'} "
          f"(weight: {loss_cfg['graph_weight']})")
    print(f"  BYOL Loss: {'✓' if loss_cfg['use_byol_loss'] else '✗'} "
          f"(weight: {loss_cfg['byol_weight']})")

    # Create callbacks
    callbacks = create_callbacks(config, exp_dir)

    # Create logger
    logger = TensorBoardLogger(
        save_dir=exp_dir,
        name='tensorboard_logs'
    )

    # Create trainer
    print("\nCreating trainer...")
    # Determine strategy based on number of devices
    if config['system']['devices'] == 1:
        strategy = 'auto'  # Use single device strategy
    else:
        strategy = 'ddp'  # Use DDP for multi-GPU

    trainer = L.Trainer(
        max_epochs=config['training']['max_epochs'],
        min_epochs=config['training']['min_epochs'],
        accelerator=config['system']['accelerator'],
        devices=config['system']['devices'],
        strategy=strategy,
        precision=config['system']['precision'],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=config['training']['log_every_n_steps'],
        deterministic='warn',  # Use 'warn' instead of True to avoid errors with bicubic interpolation
        accumulate_grad_batches=config['training']['accumulate_grad_batches'],
        gradient_clip_val=config['training']['gradient_clip_val'],
        num_sanity_val_steps=config['system']['num_sanity_val_steps'],
        check_val_every_n_epoch=config['system'].get('check_val_every_n_epoch', 1),  # Validate every N epochs
    )

    print(f"\nUsing strategy: {trainer.strategy.__class__.__name__}")

    # Train
    print("\nStarting training...")
    print("=" * 80)
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Test
    print("\nTesting best model...")
    trainer.test(model, dataloaders=test_loader, ckpt_path='best')

    print("\nTraining completed!")
    print(f"Results saved to: {exp_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
