import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from utils.utils import Dotdict
from augmentations import DataAugmentation

from models.losses import CosineSimilarity, AsymmetricLoss
from models.layers import SAGE, Classifier

class SupervisedModel(L.LightningModule):
    """
    A LightningModule for supervised multi-label classification (MLC).
    """
    def __init__(self, config, backbone, num_leaves, learning_task = 'mlc', edge_index=None):
        """
        Initialize the supervised MLC model.
        Args:
            config (dict): Configuration parameters.
            backbone (nn.Module): Backbone neural network model.
            num_leaves (int): Number of output nodes in the final classifier layer.
        """
        super().__init__()
        self.backbone = backbone
        self.edge_index = edge_index
        self.fc = Classifier(backbone.num_classes, backbone.num_classes)
        self.loss_cls = nn.BCEWithLogitsLoss()
        self.num_leaves = num_leaves
        self.learning_task = learning_task
        

        self.train_transforms = DataAugmentation(mode='weak')

        # Save hyperparameters
        training_params = config.training.to_dict() if isinstance(config.training, Dotdict) else config.training
        dataset_params = config.dataset.to_dict() if isinstance(config.dataset, Dotdict) else config.dataset
        hparams = {**training_params, **dataset_params}
        self.save_hyperparameters(hparams)

        print(f"\n\nNUMBER OF LEAVES IS: {self.num_leaves} FOR TASK: {learning_task}")
        

    def _compute_cls_loss(self, x_cls, y):
        """Compute the classifier loss."""
        embeddings = x_cls.mean(dim=-1)
        logits = self.fc(embeddings)[:len(y)]
        return logits, self.loss_cls(logits, y)

    def _prepare_data(self, x, u=None):
        """Prepare data with appropriate augmentations."""
        x = self.train_transforms(x)  # Weak augmentation for labeled data
        return x
        
    def forward(self, x, y):
        """Forward pass through the backbone and classifier."""
        x = self._prepare_data(x)
        x_cls, _ = self.backbone(x)
        logits, loss = self._compute_cls_loss(x_cls, y)
        
        return {
            'logits': logits[:, :self.num_leaves],
            'loss': loss,
        }

    def training_step(self, batch, batch_idx):
        """Compute and return the training loss."""
        x, y = batch['x'], batch['one_hot'] if self.learning_task == 'mlc' else batch['h_one_hot']
        # Forward pass
        outputs = self.forward(x, y)
        loss = outputs['loss']
        
        # Log the loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Compute and log the validation loss."""
        x, y = batch['x'], batch['one_hot'] if self.learning_task == 'mlc' else batch['h_one_hot']
        loss = self.forward(x, y)['loss']
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Perform the prediction and return logits and embeddings."""
        x, y = batch['x'], batch['one_hot'] if self.learning_task == 'mlc' else batch['h_one_hot']
        outputs = self.forward(x, y)

        outputs = {
            'logits': outputs['logits'][:, :self.num_leaves],
            'labels': y[:, :self.num_leaves],
            'embeddings': self.backbone(x)[1]
        }

        return outputs

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Gather all model parameters to include in the optimizer
        parameters = list(self.backbone.parameters()) + list(self.fc.parameters())

        # Initialize the optimizer
        optimizer = torch.optim.AdamW(
            parameters, 
            lr=self.hparams.lr, 
            betas=(0.9, 0.95)
        )

        optimizer_config = {
            "optimizer": optimizer,
        }

        # Configure learning rate scheduler if applicable
        if self.hparams.apply_scheduler:
            if not self.trainer:
                raise ValueError("Trainer not initialized. Scheduler requires a trainer instance.")
            
            print(f"\nNumber of batches {self.trainer.estimated_stepping_batches}")
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.max_lr, 
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=0.2,
            )

            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #     factor=0.5, patience=10,
            #     optimizer=optimizer, mode='min'
            #     )
            
            optimizer_config["lr_scheduler"] = {
                "scheduler": scheduler,
                "name": 'train/lr',
                "interval": "step",
                "monitor": "val_loss",
                "frequency": 1
            }

        return optimizer_config