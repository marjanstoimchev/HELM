#!/bin/bash

# Test script for all HELM training modes
# This script tests all configurations with minimal epochs for validation

DATASET="dfc_15"
BATCH_SIZE=8
MAX_EPOCHS=2
DEVICES=1
GPU=0

echo "=============================================="
echo "Testing All HELM Training Modes"
echo "=============================================="
echo "Dataset: $DATASET"
echo "Batch size: $BATCH_SIZE"
echo "Max epochs: $MAX_EPOCHS (for testing)"
echo "GPU: $GPU"
echo "=============================================="
echo ""

# 1. MLC Mode (supervised_only) - leaf labels only
echo "[1/5] Testing MLC mode (supervised_only)..."
echo "Mode: Multi-Label Classification (8 leaf classes only)"
CUDA_VISIBLE_DEVICES=$GPU python train.py \
    dataset=$DATASET \
    training=supervised_only \
    training.max_epochs=$MAX_EPOCHS \
    training.batch_size=$BATCH_SIZE \
    system.devices=$DEVICES \
    training.experiment_name=test_mlc

if [ $? -eq 0 ]; then
    echo "✓ MLC mode test PASSED"
else
    echo "✗ MLC mode test FAILED"
    exit 1
fi
echo ""

# 2. HMLC_flat Mode (supervised) - full hierarchy, no graph
echo "[2/5] Testing HMLC_flat mode..."
echo "Mode: Hierarchical Multi-Label Classification (18 classes, no graph)"
CUDA_VISIBLE_DEVICES=$GPU python train.py \
    dataset=$DATASET \
    training=hmlc_flat \
    training.max_epochs=$MAX_EPOCHS \
    training.batch_size=$BATCH_SIZE \
    system.devices=$DEVICES \
    training.experiment_name=test_hmlc_flat

if [ $? -eq 0 ]; then
    echo "✓ HMLC_flat mode test PASSED"
else
    echo "✗ HMLC_flat mode test FAILED"
    exit 1
fi
echo ""

# 3. HMLC + Graph Mode (supervised) - full hierarchy with graph learning
echo "[3/5] Testing HMLC + Graph mode..."
echo "Mode: HMLC with Graph Learning (18 classes + graph)"
CUDA_VISIBLE_DEVICES=$GPU python train.py \
    dataset=$DATASET \
    training=graph_only \
    training.max_epochs=$MAX_EPOCHS \
    training.batch_size=$BATCH_SIZE \
    system.devices=$DEVICES \
    training.experiment_name=test_hmlc_graph

if [ $? -eq 0 ]; then
    echo "✓ HMLC + Graph mode test PASSED"
else
    echo "✗ HMLC + Graph mode test FAILED"
    exit 1
fi
echo ""

# 4. HMLC_flat + BYOL Mode (semi-supervised) - full hierarchy with BYOL
echo "[4/5] Testing HMLC_flat + BYOL mode (semi-supervised)..."
echo "Mode: HMLC with BYOL (18 classes + BYOL, semi-supervised)"
CUDA_VISIBLE_DEVICES=$GPU python train.py \
    dataset=$DATASET \
    training=byol_ssl \
    training.max_epochs=$MAX_EPOCHS \
    training.batch_size=$BATCH_SIZE \
    system.devices=$DEVICES \
    training.experiment_name=test_hmlc_byol

if [ $? -eq 0 ]; then
    echo "✓ HMLC_flat + BYOL mode test PASSED"
else
    echo "✗ HMLC_flat + BYOL mode test FAILED"
    exit 1
fi
echo ""

# 5. HMLC + Graph + BYOL Mode (semi-supervised) - full mode
echo "[5/5] Testing HMLC + Graph + BYOL mode (semi-supervised)..."
echo "Mode: Full HMLC (18 classes + graph + BYOL, semi-supervised)"
CUDA_VISIBLE_DEVICES=$GPU python train.py \
    dataset=$DATASET \
    training=full_ssl \
    training.max_epochs=$MAX_EPOCHS \
    training.batch_size=$BATCH_SIZE \
    system.devices=$DEVICES \
    training.experiment_name=test_hmlc_full

if [ $? -eq 0 ]; then
    echo "✓ HMLC + Graph + BYOL mode test PASSED"
else
    echo "✗ HMLC + Graph + BYOL mode test FAILED"
    exit 1
fi
echo ""

echo "=============================================="
echo "All tests completed successfully!"
echo "=============================================="
echo ""
echo "Summary of tested modes:"
echo "1. MLC (supervised_only): 8 leaf classes, classification loss only"
echo "2. HMLC_flat: 18 classes, classification loss only"
echo "3. HMLC + Graph: 18 classes, classification + graph loss"
echo "4. HMLC_flat + BYOL: 18 classes, classification + BYOL (semi-supervised)"
echo "5. HMLC + Graph + BYOL: 18 classes, all losses (semi-supervised)"
echo ""
echo "Note: All modes use leaf-only predictions for evaluation (fair comparison)"
