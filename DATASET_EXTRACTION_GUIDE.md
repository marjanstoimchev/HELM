# Dataset Extraction Guide

Extract HELM datasets to CSV format with hierarchical labels for use with external frameworks and tools.

## Overview

The extraction script (`extract_dataset.py`) converts datasets into standardized CSV format containing:

- **image_path**: Relative image filename (no root directory)
- **split**: Dataset split assignment (train/val/test)
- **Label columns**: Binary vectors (0/1) for all hierarchy nodes

The script automatically:
- Loads hierarchy structure and orders labels topologically
- Computes non-leaf labels from descendant leaves
- Splits dataset (70% train, 15% val, 15% test)
- Sanitizes column names for CSV compatibility
- Creates label mapping file for reference

## Quick Start

```bash
# Basic usage (uses registered default paths)
python extract_dataset.py --dataset DATASET_NAME --output OUTPUT_FILE.csv

# Override data directory for local datasets
python extract_dataset.py --dataset DATASET_NAME --output OUTPUT_FILE.csv --data-dir /path/to/data
```

## Examples

### Remote Sensing Datasets

```bash
# DFC-15 (multi-label, 8 leaf classes, 18 total with hierarchy)
python extract_dataset.py --dataset dfc_15 --output datasets/dfc_15.csv

# AID (30 aerial scene classes)
python extract_dataset.py --dataset aid --output datasets/aid.csv

# UC Merced (21 land use classes)
python extract_dataset.py --dataset ucm --output datasets/ucm.csv

# MLRSNet (multi-label remote sensing)
python extract_dataset.py --dataset mlrsnet --output datasets/mlrsnet.csv
```

### Fine-Grained Classification

```bash
# CUB-200-2011 (bird species) - uses registered default path
python extract_dataset.py --dataset cub200 --output datasets/cub200.csv

# Stanford Cars (HuggingFace - auto-downloaded)
python extract_dataset.py --dataset stanford_cars --output datasets/stanford_cars.csv

# Oxford-IIIT Pets - uses registered default path
python extract_dataset.py --dataset oxford_pets --output datasets/oxford_pets.csv

# Override data directory if needed
python extract_dataset.py --dataset cub200 --output datasets/cub200.csv --data-dir /custom/path/to/cub200
```

### Medical Imaging

```bash
# ChestX-ray8
python extract_dataset.py --dataset chestxray8 --output datasets/chestxray8.csv

# PadChest
python extract_dataset.py --dataset padchest --output datasets/padchest.csv
```

## Output Format

### Main CSV File

The output CSV contains:

| Column | Type | Description |
|--------|------|-------------|
| image_path | string | Relative filename (e.g., "image001.png") |
| split | string | "train", "val", or "test" |
| [label_1] | int | Binary label (0 or 1) |
| [label_2] | int | Binary label (0 or 1) |
| ... | ... | ... |

### Label Columns

Labels are ordered hierarchically:
1. **Non-leaf nodes** (parent categories): Ordered topologically
2. **Leaf nodes** (specific classes): Original order

Non-leaf labels are automatically computed:
- Value is 1 if ANY descendant leaf is 1
- Value is 0 if ALL descendant leaves are 0

### Column Name Sanitization

Original label names are sanitized for CSV compatibility:
- Special characters replaced with underscore
- Converted to lowercase
- Multiple underscores collapsed to single

Example:
- Original: `1. Artificial surfaces`
- Sanitized: `1_artificial_surfaces`

### Label Mapping File

A companion file `*_label_mapping.csv` is created with:

```csv
sanitized_name,original_name
1_artificial_surfaces,1. Artificial surfaces
3_forest_and_semi_natural_aryeeas,3. Forest and semi-natural aryeeas
...
```

## Example Output

### DFC-15 Dataset

```csv
image_path,split,1_artificial_surfaces,3_forest_and_semi...,impervious,water,building,...
1.png,val,1,0,1,1,0,...
10.png,train,1,0,1,0,1,...
100.png,train,0,1,0,0,0,...
```

**Statistics**:
- Total samples: 3342
- Columns: 20 (image_path, split, 18 labels)
- Splits: 2104 train, 569 val, 669 test

## Using Extracted Datasets

### PyTorch Example

```python
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch

class CSVDataset(Dataset):
    def __init__(self, csv_path, image_root, split='train', transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split]
        self.image_root = image_root
        self.transform = transform

        # Extract label columns
        self.label_cols = [col for col in self.df.columns
                          if col not in ['image_path', 'split']]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load image
        img_path = self.image_root / row['image_path']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Get labels
        labels = torch.tensor(row[self.label_cols].values, dtype=torch.float32)

        return image, labels

# Usage
from torchvision import transforms
from pathlib import Path

dataset = CSVDataset(
    csv_path='datasets/dfc_15.csv',
    image_root=Path('/path/to/images'),
    split='train',
    transform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
)

from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

### Analysis Example

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('datasets/dfc_15.csv')

# Get label columns
label_cols = [col for col in df.columns if col not in ['image_path', 'split']]

# Compute label statistics
label_counts = df[label_cols].sum().sort_values(ascending=False)

# Plot distribution
fig, ax = plt.subplots(figsize=(15, 6))
label_counts.plot(kind='bar', ax=ax)
ax.set_title('Label Distribution')
ax.set_xlabel('Labels')
ax.set_ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('label_distribution.png')

# Print statistics
print(f"Total samples: {len(df)}")
print(f"Number of labels: {len(label_cols)}")
print(f"\nSplit distribution:")
print(df['split'].value_counts())
print(f"\nTop 10 most frequent labels:")
print(label_counts.head(10))
```

## Dataset Requirements

### Configuration Files

Each dataset requires:

**Hierarchy config**: `configs/dataset/hierarchies/{dataset_name}.yaml`
- Nested dictionary defining label hierarchy
- Leaf labels list

### Data Access

**HuggingFace Datasets** (AID, UCM, MLRSNet, Stanford Cars):
- Automatically downloaded on first use
- No manual setup required
- Requires internet connection
- `data_dir` is automatically set to None

**Local Datasets** (DFC-15, CUB-200, Oxford Pets, medical datasets):
- Each dataset has a registered default `data_dir` in its implementation
- Override with `--data-dir /path/to/data` if your data is elsewhere
- Ensure dataset is downloaded and extracted at the specified path
- Images should be accessible at the data directory

## Batch Extraction

Extract multiple datasets:

```bash
#!/bin/bash
# extract_all.sh

mkdir -p datasets

for dataset in dfc_15 aid ucm cub200 stanford_cars; do
    echo "Extracting $dataset..."
    python extract_dataset.py \
        --dataset $dataset \
        --output datasets/${dataset}.csv
done

echo "All datasets extracted!"
```

## Verification

Verify extracted CSV:

```python
import pandas as pd

# Load CSV
df = pd.read_csv('datasets/dfc_15.csv')

# Check structure
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Check splits
print(f"\nSplit distribution:")
print(df['split'].value_counts())

# Check for missing values
assert df.isnull().sum().sum() == 0, "Missing values found!"

# Check label columns
label_cols = [col for col in df.columns if col not in ['image_path', 'split']]
print(f"\nNumber of labels: {len(label_cols)}")

# Verify binary values
for col in label_cols:
    unique_vals = df[col].unique()
    assert set(unique_vals).issubset({0, 1}), f"Non-binary values in {col}"

print("\n✓ All checks passed!")
```

## Troubleshooting

### Missing Hierarchy Configuration

**Error**: `FileNotFoundError: Hierarchy config not found`

**Solution**: Ensure hierarchy config exists:
```bash
ls configs/dataset/hierarchies/your_dataset.yaml
```

### Missing Data Directory

**Error**: `FileNotFoundError: No such file or directory: '../path/to/dataset'`

**Solution**: The dataset's registered default path doesn't exist on your system. Override with:
```bash
python extract_dataset.py --dataset DATASET_NAME --output output.csv --data-dir /your/actual/path
```

### HuggingFace Access

**Error**: `datasets` module not found

**Solution**: Install HuggingFace datasets:
```bash
pip install datasets
```

### Memory Issues

For large datasets, process in chunks:

```python
# Load in chunks
chunks = []
for chunk in pd.read_csv('large_dataset.csv', chunksize=10000):
    # Process chunk
    chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
```

## Command Line Options

```bash
python extract_dataset.py [OPTIONS]

Required:
  --dataset DATASET_NAME       Name of the dataset to extract
  --output OUTPUT_FILE         Path to output CSV file

Optional:
  --data-dir DATA_DIR          Override data directory for local datasets
  --config-dir CONFIG_DIR      Configuration directory (default: configs)
```

### Examples with Options

```bash
# Use custom data directory
python extract_dataset.py \
    --dataset cub200 \
    --output datasets/cub200.csv \
    --data-dir /mnt/storage/datasets/cub200

# Use custom config directory
python extract_dataset.py \
    --dataset dfc_15 \
    --output datasets/dfc_15.csv \
    --config-dir /path/to/configs
```

### Integration with Other Frameworks

**TensorFlow/Keras**:
```python
import pandas as pd
import tensorflow as tf

df = pd.read_csv('datasets/dfc_15.csv')
# Convert to TensorFlow Dataset
# ... (implementation specific)
```

**scikit-learn**:
```python
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('datasets/dfc_15.csv')
train_df = df[df['split'] == 'train']

X = train_df['image_path'].values
y = train_df[label_cols].values
# ... (continue with sklearn)
```

## Output Summary

After extraction, you will have:

```
datasets/
├── dfc_15.csv                  # Main dataset CSV
├── dfc_15_label_mapping.csv    # Label name mapping
├── aid.csv
├── aid_label_mapping.csv
└── ...
```

Each CSV provides a clean, standardized format for training models in any framework while preserving the hierarchical label structure.
