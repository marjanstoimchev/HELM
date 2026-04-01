import os
import gc
import glob
import hydra
from hydra.utils import get_original_cwd
import torch
import lightning as L
from lightning import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from omegaconf import DictConfig, OmegaConf

from data.dataset_pipeline import DatasetPipeline
from data.hierarchy import create_edge_index
from datamodules.base_datamodule import BaseDataModule
from models.model import h_deit_base_embedding
from augmentations import Preprocess
from callbacks import ModelCheckpoint_, EarlyStopping_, RichProgressBar_
from utils.utils import Dotdict, calculate_metrics, predict
from trainers import get_trainer_class


def resolve_method_name(cfg):
    """Build the method directory name from config (e.g. 'hmlc-ssl-graph-byol')."""
    parts = [cfg.method.learning_task, cfg.method.training_mode]
    if cfg.method.learning_task == 'hmlc' and cfg.method.apply_edges:
        parts.append('graph')
    if cfg.method.apply_byol:
        parts.append('byol')
    return '-'.join(parts)


def get_batch_size(cfg, fraction):
    """Look up batch size from training config."""
    frac_key = int(100 * fraction) if fraction is not None else 100
    batch_size_map = OmegaConf.to_container(cfg.training.batch_size_map)
    return batch_size_map.get(frac_key, 16)


def run_experiment(cfg, seed, fraction):
    """Run a single training + inference experiment."""
    frac_int = int(100 * fraction) if fraction is not None else 100
    frac_str = f"fraction_{frac_int}"

    config = Dotdict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    seed_everything(seed, workers=True)

    # Resolve paths relative to the original working directory (Hydra changes cwd)
    orig_cwd = get_original_cwd()
    yaml_path = os.path.join(orig_cwd, f'configs/dataset/{cfg.dataset.folder_name}.yaml')
    cache_dir = os.path.join(orig_cwd, cfg.data.cache_dir) if not os.path.isabs(cfg.data.cache_dir) else cfg.data.cache_dir

    # Data pipeline
    pipeline = DatasetPipeline(
        yaml_path=yaml_path,
        seed=seed,
        cache_dir=cache_dir,
    )
    # In supervised mode, use all data (no labeled/unlabeled split)
    effective_fraction = fraction if cfg.method.training_mode == 'ssl' else None
    outputs = pipeline.run_pipeline(effective_fraction)

    # In supervised mode, remove unlabeled data (if any)
    if cfg.method.training_mode == 'sl':
        outputs.pop('U', None)

    # Resolve model dimensions
    learning_task = cfg.method.learning_task
    num_leaves = pipeline.num_classes
    num_classes = pipeline.num_classes if learning_task == 'mlc' else pipeline.num_classes_extended

    # Edge index for graph methods
    edge_index = None
    if cfg.method.apply_edges and learning_task == 'hmlc':
        edge_index = create_edge_index(hierarchy=pipeline.label_to_predecessors)

    # Batch size & datamodule
    batch_size = get_batch_size(cfg, fraction)
    transforms = Preprocess()
    datamodule = BaseDataModule(
        outputs,
        batch_size=batch_size,
        num_workers=cfg.processing.num_workers,
        transforms=transforms,
    )

    # Method name & paths
    method_name = resolve_method_name(cfg)
    save_dir = os.path.join(orig_cwd, cfg.experiment.save_model_dir, cfg.dataset.folder_name, method_name, frac_str, f"seed_{seed}")
    results_dir = os.path.join(orig_cwd, cfg.experiment.output_dir, cfg.dataset.folder_name, method_name, frac_str)

    print(f"\n{'='*60}")
    print(f"Dataset: {cfg.dataset.folder_name} | Method: {method_name}")
    print(f"Fraction: {frac_int}% | Seed: {seed} | Batch size: {batch_size}")
    print(f"Classes: {num_classes} (leaves: {num_leaves})")
    print(f"{'='*60}\n")

    # Build model
    backbone = h_deit_base_embedding(num_classes=num_classes, pretrained=cfg.model.pretrained)
    trainer_cls = get_trainer_class(
        cfg.method.training_mode, learning_task,
        cfg.method.apply_edges, cfg.method.apply_byol,
    )
    lightning_model = trainer_cls(config, backbone, num_leaves, learning_task, edge_index)

    # Callbacks
    callbacks = [
        LearningRateMonitor(logging_interval='step'),
        RichProgressBar_(),
        EarlyStopping_(metric="val_loss", mode="min", patience=cfg.training.patience),
        ModelCheckpoint_(dirpath=save_dir, metric="val_loss", mode="min"),
    ]

    # Train
    trainer = L.Trainer(
        enable_model_summary=True,
        num_sanity_val_steps=0,
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.strategy,
        devices=list(cfg.trainer.devices),
        precision=cfg.trainer.precision,
        min_epochs=cfg.training.min_epochs,
        max_epochs=cfg.training.epochs,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        sync_batchnorm=cfg.trainer.sync_batchnorm,
        benchmark=True,
        log_every_n_steps=cfg.training.log_every_n_steps,
        enable_checkpointing=True,
        callbacks=callbacks,
    )
    trainer.fit(lightning_model, datamodule=datamodule)

    del trainer, backbone, lightning_model

    # Inference — load the best checkpoint saved during training
    ckpt_path = glob.glob(os.path.join(save_dir, '*.ckpt'))[0]
    os.makedirs(results_dir, exist_ok=True)

    backbone_inf = h_deit_base_embedding(num_classes=num_classes, pretrained=cfg.model.pretrained)
    model_inf = trainer_cls.load_from_checkpoint(
        ckpt_path, config=config, backbone=backbone_inf,
        num_leaves=num_classes, learning_task=learning_task, edge_index=edge_index,
    )

    trainer_inf = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.strategy,
        devices=list(cfg.trainer.devices),
        callbacks=callbacks,
    )
    Y = predict(trainer_inf, model_inf, datamodule)
    df_metrics = calculate_metrics(Y)
    df_metrics = df_metrics.rename(columns={0: f"seed_{seed}"})
    df_metrics.to_csv(os.path.join(results_dir, f"metrics_seed_{seed}.txt"), sep="\t")

    del trainer_inf, model_inf
    gc.collect()


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    for seed in cfg.experiment.seeds:
        for fraction in cfg.experiment.fractions:
            run_experiment(cfg, seed, fraction)


if __name__ == "__main__":
    main()
