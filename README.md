# HELM: Hierarchical and Explicit Label Modeling

[![Tests](https://github.com/marjanstoimchev/HELM/actions/workflows/tests.yml/badge.svg)](https://github.com/marjanstoimchev/HELM/actions/workflows/tests.yml)

A framework for **hierarchical multi-label image classification** that combines Vision Transformers with graph-based label propagation and self-supervised learning (BYOL). Supports supervised and semi-supervised training across remote sensing, medical imaging, and microscopy datasets.

---

## Key Ideas

- **Per-label classification tokens** — instead of a single CLS token, the ViT backbone uses one classification token per label, enabling label-specific attention
- **Graph-based label propagation** — a GraphSAGE module operates over the label hierarchy to propagate information between parent and child labels
- **BYOL self-supervision** — Bootstrap Your Own Latent learns representations from unlabeled data alongside the supervised objective
- **Semi-supervised training** — train with as little as 1% labeled data by combining labeled supervision with unlabeled BYOL

---

## Quick Start

### 1. Create the environment

```bash
conda create -n helm python=3.11 -y
conda activate helm
pip install -r requirements.txt
```

> **Note**: PyTorch Geometric may require separate installation depending on your CUDA version.
> See [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### 2. Train

```bash
# Default: UCM dataset, full HELM method, 4 GPUs
python train.py
```

### 3. Results

Training iterates over all `seeds x fractions` and produces:
```
saved_models/{dataset}/{method}/fraction_{pct}/seed_{seed}/   # checkpoints
results/{dataset}/{method}/fraction_{pct}/metrics_seed_{seed}.txt   # metrics
```

---

## Method Variants

HELM is modular. Each component can be toggled independently via Hydra configs:

| Config | Hierarchy | Training | Graph | BYOL | Description |
|--------|:---------:|:--------:|:-----:|:----:|-------------|
| `mlc-sl` | - | Supervised | - | - | Flat multi-label baseline |
| `hmlc-sl` | Yes | Supervised | - | - | Hierarchical labels only |
| `hmlc-sl-graph` | Yes | Supervised | Yes | - | + graph label propagation |
| `hmlc-sl-graph-byol` | Yes | Supervised | Yes | Yes | + BYOL self-supervision |
| `hmlc-ssl-byol` | Yes | Semi-sup | - | Yes | Semi-supervised + BYOL |
| `hmlc-ssl-graph` | Yes | Semi-sup | Yes | - | Semi-supervised + graph |
| **`hmlc-ssl-graph-byol`** | **Yes** | **Semi-sup** | **Yes** | **Yes** | **Full HELM** (default) |

```bash
python train.py method=hmlc-ssl-graph-byol   # full HELM
python train.py method=mlc-sl                 # flat baseline
```

---

## Architecture

```
Input Image (224x224)
    │
    ▼
Patch Embedding (16x16 → 196 patch tokens)
    │
    ▼
Prepend M classification tokens (one per label)
    │
    ▼
12x Transformer Blocks (ViT-Base, 768-dim, 12 heads)
    │
    ├─► CLS Embeddings ──► Classifier ──► Logits
    │                         │
    │                    [if graph]
    │                         ▼
    │                  GraphSAGE (2 layers)
    │                  over hierarchy edges
    │                         │
    │                         ▼
    │                  Graph Classifier ──► Graph Logits
    │
    └─► Patch Embeddings ──► [if BYOL] ──► Contrastive Loss
                                           (momentum encoder)
```

In **semi-supervised mode**, labeled data receives weak + strong augmentations. Unlabeled data is used exclusively for the BYOL objective. Labeled data is upsampled to match the unlabeled set size.

---

## Datasets

9 datasets across three domains. See [DATASETS.md](DATASETS.md) for full hierarchies and details.

| Dataset | Domain | Classes | Hierarchy Depth | Config |
|---------|--------|:-------:|:---------------:|--------|
| UCM | Remote Sensing | 17 | 3 | `dataset=ucm` |
| AID | Remote Sensing | 17 | 3 | `dataset=aid` |
| MLRSNet | Remote Sensing | 60 | 5 | `dataset=mlrsnet` |
| DFC-15 | Remote Sensing | 8 | 3 | `dataset=dfc_15` |
| ChestX-ray8 | Medical Imaging | 20 | 3 | `dataset=chestxray8` |
| NIH ChestXray | Medical Imaging | 15 | 3 | `dataset=nihchestxray` |
| PadChest | Medical Imaging | 19 | 6 | `dataset=padchest` |
| MuReD | Medical Imaging | 20 | 4 | `dataset=mured` |
| HPA | Microscopy | 28 | 2 | `dataset=hpa` |

---

## Project Structure

```
HELM/
├── train.py                        # Main entry point
├── configs/
│   ├── config.yaml                 # Hydra defaults (dataset, method, training)
│   ├── dataset/                    # Per-dataset configs with label hierarchies
│   ├── method/                     # 7 method variants
│   └── training/                   # Hyperparameters
├── models/
│   ├── model.py                    # Hierarchical ViT backbone
│   ├── layers.py                   # SAGE, GAT, GCN, Classifier
│   ├── losses.py                   # BCE, ZLPR, Cosine, Asymmetric losses
│   └── byol.py                     # BYOL module
├── trainers/
│   ├── supervised.py               # Supervised trainer
│   ├── ssl_byol.py                 # Semi-supervised + BYOL
│   ├── ssl_graph.py                # Semi-supervised + Graph
│   └── ssl_graph_byol.py           # Full HELM (Graph + BYOL)
├── data/
│   ├── dataset_pipeline.py         # Data loading orchestrator
│   ├── hierarchy.py                # Hierarchy parsing & label extension
│   ├── dataset.py                  # Dataset classes
│   └── *.py                        # Per-dataset loaders
├── datamodules/
│   └── base_datamodule.py          # Lightning DataModule
├── augmentations.py                # Kornia weak/strong augmentations
├── callbacks.py                    # Checkpoint, early stopping, progress bar
├── utils/
│   └── utils.py                    # Metrics, utilities
├── scripts/
│   ├── measure_all_variants.py     # Benchmark params/throughput/memory
│   └── token_ablation.py           # Ablation studies
├── tests/                          # Test suite (339 tests)
├── DATASETS.md                     # Dataset details and hierarchies
└── requirements.txt
```

---

## Usage Reference

All configuration uses [Hydra](https://hydra.cc/). Every parameter below can be overridden from the command line.

### Selecting Dataset and Method

```bash
# Dataset (one of: ucm, aid, mlrsnet, dfc_15, chestxray8, nihchestxray, padchest, mured, hpa)
python train.py dataset=chestxray8

# Method variant (see Method Variants table above)
python train.py method=hmlc-ssl-graph-byol    # full HELM (default)
python train.py method=mlc-sl                  # flat baseline, no hierarchy
python train.py method=hmlc-sl                 # hierarchical, supervised only
python train.py method=hmlc-ssl-graph          # semi-supervised + graph, no BYOL
python train.py method=hmlc-ssl-byol           # semi-supervised + BYOL, no graph
```

### GPU and Hardware

```bash
# Single GPU
python train.py trainer.devices=[0]

# Specific GPUs
python train.py trainer.devices=[2,3]

# All 8 GPUs
python train.py trainer.devices=[0,1,2,3,4,5,6,7]

# Full precision (default: 16-mixed)
python train.py trainer.precision=32

# Disable sync batchnorm (for single GPU)
python train.py trainer.sync_batchnorm=false

# DDP strategy (default)
python train.py trainer.strategy=ddp_find_unused_parameters_true

# CPU only (for debugging)
python train.py trainer.accelerator=cpu trainer.devices=1
```

### Semi-Supervised: Labeled Data Fractions

The `experiment.fractions` parameter controls what percentage of training data is labeled.
The rest becomes unlabeled data for BYOL. Training loops over all seeds x fractions.

```bash
# Default: 1%, 5%, 10%, 25% labeled data
python train.py experiment.fractions=[0.01,0.05,0.1,0.25]

# Single fraction (e.g., 10% labeled)
python train.py experiment.fractions=[0.1]

# Fully supervised (100% labeled, no unlabeled split)
python train.py experiment.fractions=[1.0]

# Very low-label regime
python train.py experiment.fractions=[0.01]
```

> **Note**: In supervised methods (`sl`), the unlabeled split is discarded. Fractions only affect the training set size.
> In semi-supervised methods (`ssl`), the unlabeled split is used for BYOL.

### Seeds and Reproducibility

```bash
# Default: 3 seeds
python train.py experiment.seeds=[0,1,42]

# Quick single-seed run
python train.py experiment.seeds=[42]

# Training is deterministic by default
python train.py training.deterministic=true
```

### Batch Size

Batch size is automatically selected from the fraction via `training.batch_size_map`:

| Fraction | 1% | 5% | 10% | 25% | 50% | 75% | 100% |
|----------|:--:|:--:|:---:|:---:|:---:|:---:|:----:|
| Batch size | 4 | 16 | 16 | 16 | 32 | 32 | 16 |

Override specific entries:

```bash
# Override the batch size for 10% fraction
python train.py training.batch_size_map.10=32

# Override for 1% fraction
python train.py training.batch_size_map.1=8
```

### Learning Rate and Scheduler

```bash
# Learning rate
python train.py training.lr=5e-5 training.head_lr=5e-5

# OneCycleLR peak learning rate
python train.py training.max_lr=1e-3

# Disable the LR scheduler entirely
python train.py training.apply_scheduler=false
```

### Training Duration and Early Stopping

```bash
# Max and min epochs
python train.py training.epochs=200 training.min_epochs=10

# Early stopping patience (epochs without val_loss improvement)
python train.py training.patience=10

# Gradient accumulation (effective batch = batch_size * accumulate)
python train.py training.accumulate_grad_batches=10
```

### Model

```bash
# Disable pretrained ViT weights (train from scratch)
python train.py model.pretrained=false

# Enable pretrained weights (default)
python train.py model.pretrained=true
```

### Data Loading

```bash
# Number of dataloader workers
python train.py processing.num_workers=8

# Dataset cache directory (preprocessed .npy files are stored here)
# On first run, images are downloaded and saved as .npy for fast reuse
python train.py data.cache_dir=./Datasets/mlc_datasets_npy          # default
python train.py data.cache_dir=/scratch/datasets/helm_cache          # custom path
```

### Output Directories

```bash
# Where to save checkpoints and metrics
python train.py experiment.output_dir=my_results experiment.save_model_dir=my_models
```

### Full Example

```bash
# Train HELM on ChestX-ray8 with 10% labels, single GPU, 50 epochs
python train.py \
    dataset=chestxray8 \
    method=hmlc-ssl-graph-byol \
    trainer.devices=[0] \
    experiment.fractions=[0.1] \
    experiment.seeds=[42] \
    training.epochs=50 \
    training.lr=5e-5 \
    training.patience=10

# Supervised baseline on UCM, all data, 2 GPUs
python train.py \
    dataset=ucm \
    method=mlc-sl \
    trainer.devices=[0,1] \
    experiment.fractions=[1.0] \
    experiment.seeds=[0,1,42]

# Quick debug run on CPU
python train.py \
    dataset=ucm \
    method=hmlc-sl \
    trainer.accelerator=cpu \
    trainer.devices=1 \
    experiment.seeds=[42] \
    experiment.fractions=[0.25] \
    training.epochs=2
```

### All Overridable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset` | `ucm` | Dataset config name (see Datasets table) |
| `method` | `hmlc-ssl-graph-byol` | Method variant (see Method Variants table) |
| `model.pretrained` | `true` | Load ImageNet-pretrained ViT-Base weights |
| `processing.num_workers` | `16` | DataLoader worker processes |
| `data.cache_dir` | `./Datasets/mlc_datasets_npy` | Where preprocessed .npy files are cached |
| `experiment.seeds` | `[0, 1, 42]` | Random seeds (trains one model per seed) |
| `experiment.fractions` | `[0.01, 0.05, 0.1, 0.25]` | Labeled data fractions |
| `experiment.output_dir` | `results` | Directory for metric files |
| `experiment.save_model_dir` | `saved_models` | Directory for checkpoints |
| `trainer.accelerator` | `gpu` | `gpu` or `cpu` |
| `trainer.strategy` | `ddp_find_unused_parameters_true` | PyTorch Lightning strategy |
| `trainer.devices` | `[0, 1, 2, 3]` | GPU device IDs |
| `trainer.precision` | `16-mixed` | `16-mixed`, `bf16-mixed`, or `32` |
| `trainer.sync_batchnorm` | `true` | Sync batch norm across GPUs |
| `training.epochs` | `100` | Maximum training epochs |
| `training.min_epochs` | `5` | Minimum training epochs |
| `training.patience` | `5` | Early stopping patience (epochs) |
| `training.lr` | `1e-4` | Backbone learning rate |
| `training.head_lr` | `1e-4` | Classification head learning rate |
| `training.max_lr` | `3e-4` | OneCycleLR peak learning rate |
| `training.apply_scheduler` | `true` | Enable OneCycleLR scheduler |
| `training.accumulate_grad_batches` | `5` | Gradient accumulation steps |
| `training.deterministic` | `true` | Deterministic training |
| `training.log_every_n_steps` | `1` | Logging frequency |
| `training.batch_size_map` | `{1:4, 5:16, 10:16, 25:16, 50:32, 75:32, 100:16}` | Fraction(%) → batch size |

---

## Evaluation Metrics

The framework computes 16 metrics on the test set:

| Category | Metrics |
|----------|---------|
| Ranking | Ranking Loss, One Error, Coverage |
| Precision-Recall | Average AUPRC, Weighted AUPRC |
| Micro-averaged | Micro F1, Micro Precision, Micro Recall |
| Macro-averaged | Macro F1, Macro Precision, Macro Recall |
| Sample-averaged | Sample F1, Sample Precision, Sample Recall |
| Exact match | Subset Accuracy, Hamming Loss |

---

## Benchmarking

Measure parameters, throughput, latency, and peak GPU memory for all method variants:

```bash
python scripts/measure_all_variants.py --gpu 0
python scripts/measure_all_variants.py --gpu 0 --datasets ucm aid dfc_15
```

---

## Testing

```bash
conda activate helm
pytest tests/ -v                    # all tests
pytest tests/ -m "not slow"         # skip e2e training tests
pytest tests/test_hierarchy.py -v   # specific module
```

---

## Adding a New Dataset

1. Define the hierarchy in `configs/dataset/your_dataset.yaml` (see [DATASETS.md](DATASETS.md#adding-a-new-dataset))
2. Create a data loader in `data/`
3. Register it in `DatasetFactory` (`data/utils.py`)
4. Run: `python train.py dataset=your_dataset`
