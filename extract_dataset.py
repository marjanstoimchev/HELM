"""
Dataset Extraction Script for HELM

This script extracts datasets into CSV format with:
- image_path: Relative path to image (no root directory)
- split: train/val/test split assignment
- label columns: Binary vectors for each class in the hierarchy (ordered)

Works with all dataset types:
- HuggingFace datasets (UCM, AID, MLRSNet, etc.)
- Custom folder datasets (DFC-15, CUB-200, etc.)
- Medical datasets (ChestX-ray8, PadChest, etc.)

Usage:
    python extract_dataset.py --dataset dfc_15 --output datasets/dfc_15.csv
    python extract_dataset.py --dataset aid --output datasets/aid.csv
    python extract_dataset.py --dataset cub200 --output datasets/cub200.csv
"""

import sys
from pathlib import Path
import pandas as pd
import argparse
import re
import hydra

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data import HierarchyManager
from src.data.dataset.utils import create_dataset
from src.data.dataset.splitter import stratified_split_dataset
from src.data.hierarchies.hierarchy_utils import (
    HierarchyGraphGenerator,
    find_all_descendant_leaves
)


def sanitize_column_name(name: str) -> str:
    """
    Sanitize column name for CSV compatibility.

    Args:
        name: Original column name

    Returns:
        Sanitized column name (alphanumeric + underscore)
    """
    # Remove special characters, keep alphanumeric and underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    # Replace multiple underscores with single
    sanitized = re.sub(r'_+', '_', sanitized)
    # Convert to lowercase for consistency
    sanitized = sanitized.lower()
    return sanitized


def create_hierarchical_labels(df: pd.DataFrame, hierarchy: dict, leaf_labels: list):
    """
    Create hierarchical label columns using the HierarchyGraphGenerator.

    Args:
        df: DataFrame with leaf label columns
        hierarchy: Hierarchy dictionary
        leaf_labels: List of leaf label names

    Returns:
        tuple: (DataFrame with all hierarchy labels, node_mapping dict, label_mapping dict)
    """
    # Create graph generator to get proper node ordering
    graph_gen = HierarchyGraphGenerator(hierarchy, leaf_labels)
    graph_output = graph_gen.generate(mode="upper_triangular")
    node_mapping = graph_output['node_mapping']

    # Get ordered nodes (this is the correct order from topological sort)
    ordered_nodes = sorted(node_mapping.keys(), key=lambda x: node_mapping[x])

    # Find all descendant leaves for non-leaf nodes
    descendant_map = find_all_descendant_leaves(hierarchy, leaf_labels)

    # Create sanitized column names and mapping
    sanitized_names = {node: sanitize_column_name(node) for node in ordered_nodes}
    label_mapping = {sanitized_names[node]: node for node in ordered_nodes}

    # Rename existing leaf columns to sanitized names
    rename_dict = {orig: sanitize_column_name(orig) for orig in leaf_labels if orig in df.columns}
    df = df.rename(columns=rename_dict)

    # Create non-leaf columns
    for node in ordered_nodes:
        sanitized_node = sanitized_names[node]

        if sanitized_node in df.columns:
            # Already exists (leaf node)
            continue

        # Non-leaf node: compute from descendant leaves
        if node in descendant_map:
            descendant_leaves = descendant_map[node]
            df[sanitized_node] = 0

            for leaf in descendant_leaves:
                leaf_sanitized = sanitize_column_name(leaf)
                if leaf_sanitized in df.columns:
                    df[sanitized_node] = df[sanitized_node] | df[leaf_sanitized]
        else:
            # No descendants (shouldn't happen in valid hierarchy)
            df[sanitized_node] = 0

    return df, ordered_nodes, sanitized_names, label_mapping


def extract_image_path(row, dataset_name: str, image_column: str) -> str:
    """
    Extract relative image path (without root directory).

    Args:
        row: DataFrame row
        dataset_name: Name of dataset
        image_column: Column containing image path or name

    Returns:
        Relative image path string
    """
    # For HuggingFace datasets with synthetic paths (hf://dataset_name/123.png)
    if 'full_path' in row and isinstance(row['full_path'], str) and row['full_path'].startswith('hf://'):
        # Extract just the filename part after the dataset name
        # Example: hf://aid/123.png -> 123.png
        return row['full_path'].split('/')[-1]

    # For HuggingFace datasets, extract filename from Image object
    if image_column == 'image' and hasattr(row[image_column], 'filename'):
        # HuggingFace Image object
        return Path(row[image_column].filename).name

    # For path-based datasets
    if image_column in ['full_path', 'path', 'image_path']:
        full_path = Path(str(row[image_column]))
        # Return just the filename
        return full_path.name

    # For datasets with image_name column
    if 'image_name' in row:
        return row['image_name']

    # Fallback: return as-is
    return str(row[image_column])


def extract_dataset_to_csv(dataset_name: str, output_path: str, config_dir: str = 'configs', data_dir: str = None):
    """
    Extract dataset to CSV format.

    Args:
        dataset_name: Name of dataset (e.g., 'dfc_15', 'aid', 'cub200')
        output_path: Path to save CSV file
        config_dir: Directory containing configuration files
        data_dir: Optional data directory path for local datasets
    """
    print("=" * 80)
    print(f"EXTRACTING DATASET: {dataset_name}")
    print("=" * 80)

    # Load Hydra config
    hydra.initialize(version_base=None, config_path=config_dir)
    cfg = hydra.compose(config_name="config", overrides=[f"dataset.name={dataset_name}"])

    dataset_cfg = cfg['dataset']

    print(f"\nDataset: {dataset_name}")
    print(f"Output: {output_path}")

    # Load hierarchy
    print("\n[1/5] Loading hierarchy...")
    hierarchy, leaf_labels = HierarchyManager.load_hierarchy_config(
        dataset_cfg['hierarchy_config_dir'],
        dataset_name
    )
    print(f"  - Leaf labels: {len(leaf_labels)}")

    # Determine data_dir based on dataset type
    # HuggingFace datasets (automatically downloaded): aid, ucm, mlrsnet, stanford_cars
    # Local datasets may have default data_dir in their registration
    huggingface_datasets = {'aid', 'ucm', 'mlrsnet', 'stanford_cars'}

    # Create dataset kwargs
    dataset_kwargs = {'leaf_labels': leaf_labels}

    if dataset_name in huggingface_datasets:
        # HuggingFace datasets don't need data_dir
        dataset_kwargs['data_dir'] = None
    elif data_dir:
        # Local dataset with CLI-provided data_dir - use it
        dataset_kwargs['data_dir'] = data_dir
    # Otherwise, don't pass data_dir - let dataset use its registered default

    # Create dataset
    print("\n[2/5] Creating dataset...")
    df = create_dataset(dataset_name, **dataset_kwargs)
    print(f"  - Total samples: {len(df)}")

    # Split dataset
    print("\n[3/5] Splitting dataset...")
    split_df = stratified_split_dataset(
        df,
        label_cols=leaf_labels,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        random_state=42
    )
    print(f"  - Train: {len(split_df[split_df['split'] == 'train'])} samples")
    print(f"  - Val: {len(split_df[split_df['split'] == 'val'])} samples")
    print(f"  - Test: {len(split_df[split_df['split'] == 'test'])} samples")

    # Determine image column
    dataset_name_normalized = dataset_cfg['name'].lower().replace('-', '_')
    huggingface_datasets = {'ucm', 'aid', 'mlrsnet', 'hf', 'stanford_cars'}
    image_column = 'image' if dataset_name_normalized in huggingface_datasets else 'full_path'

    # Extract image paths (relative, no root)
    print("\n[4/5] Extracting image paths...")
    split_df['image_path'] = split_df.apply(
        lambda row: extract_image_path(row, dataset_cfg['name'], image_column),
        axis=1
    )

    # Create hierarchical label vectors
    print("\n[5/5] Creating hierarchical label vectors...")
    split_df, ordered_nodes, sanitized_names, label_mapping = create_hierarchical_labels(
        split_df, hierarchy, leaf_labels
    )

    # Get ordered sanitized column names
    ordered_labels_sanitized = [sanitized_names[node] for node in ordered_nodes]

    print(f"  - Total labels (hierarchy): {len(ordered_labels_sanitized)}")
    print(f"  - Label order (first 5): {', '.join(ordered_labels_sanitized[:5])}...")

    # Create final DataFrame with ordered columns
    final_columns = ['image_path', 'split'] + ordered_labels_sanitized
    output_df = split_df[final_columns].copy()

    # Save to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)

    # Save label mapping for reference
    mapping_path = output_path.parent / (output_path.stem + '_label_mapping.csv')
    mapping_df = pd.DataFrame([
        {'sanitized_name': k, 'original_name': v}
        for k, v in label_mapping.items()
    ])
    mapping_df.to_csv(mapping_path, index=False)

    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE!")
    print("=" * 80)
    print(f"\nSaved to: {output_path}")
    print(f"Label mapping saved to: {mapping_path}")
    print(f"Total samples: {len(output_df)}")
    print(f"Columns: {len(output_df.columns)} (image_path, split, {len(ordered_labels_sanitized)} labels)")

    # Show sample
    print("\nSample rows:")
    print(output_df.head(3).to_string())

    # Show label statistics
    print("\nLabel statistics (sanitized names):")
    label_counts = output_df[ordered_labels_sanitized].sum().sort_values(ascending=False)
    print(label_counts.head(10).to_string())

    # Cleanup Hydra
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    return output_df


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract HELM dataset to CSV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python extract_dataset.py --dataset dfc_15 --output datasets/dfc_15.csv
  python extract_dataset.py --dataset aid --output datasets/aid.csv
  python extract_dataset.py --dataset cub200 --output datasets/cub200.csv

Available datasets:
  - dfc_15, aid, ucm, mlrsnet (remote sensing)
  - cub200, stanford_cars, oxford_pets, fgvc (fine-grained)
  - chestxray8, padchest, ethec, hpa (medical)
        """
    )

    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--config-dir', type=str, default='configs', help='Configuration directory')
    parser.add_argument('--data-dir', type=str, default=None, help='Data directory path (for local datasets)')

    args = parser.parse_args()

    try:
        extract_dataset_to_csv(
            dataset_name=args.dataset,
            output_path=args.output,
            config_dir=args.config_dir,
            data_dir=args.data_dir
        )
        print("\n✅ Success!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
