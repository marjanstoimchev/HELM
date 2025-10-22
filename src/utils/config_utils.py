"""Configuration utilities for HELM."""

import os
import random
import numpy as np
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file using OmegaConf.

    Args:
        config_path: Path to config file. If None, uses default config.

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = "configs/config.yaml"

    cfg = OmegaConf.load(config_path)
    return OmegaConf.to_container(cfg, resolve=True)


def setup_experiment(config: Dict[str, Any]) -> Path:
    """
    Setup experiment directory and set random seeds.

    Args:
        config: Configuration dictionary

    Returns:
        Path to experiment directory
    """
    # Set random seeds
    seed = config['system']['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Set deterministic mode
    if config['system']['deterministic']:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create experiment directory
    save_dir = Path(config['training']['save_dir'])
    experiment_name = config['training']['experiment_name']
    exp_dir = save_dir / experiment_name

    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_save_path = exp_dir / 'config.yaml'
    OmegaConf.save(config, config_save_path)

    return exp_dir


def get_loss_config_name(loss_config: Dict[str, Any]) -> str:
    """
    Generate a descriptive name based on loss configuration.

    Args:
        loss_config: Loss configuration dictionary

    Returns:
        Descriptive name string
    """
    components = []

    if loss_config['use_classification_loss']:
        components.append('cls')
    if loss_config['use_graph_loss']:
        components.append('graph')
    if loss_config['use_byol_loss']:
        components.append('byol')

    if not components:
        return 'no_loss'

    return '_'.join(components)
