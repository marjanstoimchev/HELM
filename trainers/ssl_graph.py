import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from utils.utils import Dotdict
from augmentations import DataAugmentation
from models.layers import SAGE, Classifier

class GraphBasedModel(L.LightningModule):
    """
    A LightningModule for graph-based learning using SAGE.
    """

    def __init__(self, config, backbone, num_leaves, learning_task='hmlc', edge_index=None):
        """
        Initialize the graph-based model with SAGE.
        Args:
            config (dict): Configuration parameters.
            backbone (nn.Module): Backbone neural network model.
            num_leaves (int): Number of output nodes in the final classifier layer.
            edge_index (Tensor): Edge index for graph-based learning.
        """
        super().__init__()
        self.backbone = backbone
        self.fc = Classifier(backbone.num_classes, backbone.num_classes)
        self.sage = SAGE(backbone.embed_dim, 64, backbone.num_classes, backbone.num_classes)

        self.loss_cls = nn.BCEWithLogitsLoss()
        self.loss_gcn = nn.BCEWithLogitsLoss()
        self.train_transforms = DataAugmentation(mode='weak')

        self.edge_index = edge_index
        self.num_leaves = num_leaves

        # Save hyperparameters
        training_params = config.training.to_dict() if isinstance(config.training, Dotdict) else config.training
        dataset_params = config.dataset.to_dict() if isinstance(config.dataset, Dotdict) else config.dataset
        hparams = {**training_params, **dataset_params}
        self.save_hyperparameters(hparams)

    def _prepare_data(self, x, u=None):
        """Prepare data with appropriate augmentations based on the presence of graph data and the mode."""
        x0_l = self.train_transforms(x)  # Weak augmentation for labeled data
                
        if u is not None:  # Semi-supervised mode with unlabeled data
            x0_u = self.train_transforms(u)
            x0 = torch.cat((x0_l, x0_u), dim=0)  # Combine weak augmentations            
            return x0, None

        return x0_l, None  # Semi-supervised without unlabeled data
    
    def _compute_cls_loss(self, x_cls, y):
        """Compute the classifier loss."""
        embeddings = x_cls.mean(dim=-1)
        logits = self.fc(embeddings)[:len(y)]
        return logits, self.loss_cls(logits, y)

    def _compute_gcn_loss(self, x_cls, y):
        """Compute the GCN loss if graph data is used."""
        _, logits = self.sage(x_cls, self.edge_index.to(x_cls.device))
        return logits[:len(y)], self.loss_gcn(logits[:len(y)], y)

    def forward(self, x, y, u=None):
        """Forward pass through the backbone and classifier."""
        x, _ = self._prepare_data(x, u)
        x_cls = self.backbone(x)[0]

        logits_cls, loss_cls = self._compute_cls_loss(x_cls, y)
        logits_gcn, loss_gcn = self._compute_gcn_loss(x_cls, y)

        return {
            'logits_h': logits_cls + logits_gcn,
            'logits': logits_cls[:, :self.num_leaves] + logits_gcn[:, :self.num_leaves],  # Sum of classifier and GCN logits
            'loss': loss_cls + loss_gcn,  # Total loss
            'x_cls': x_cls,
        }

    def training_step(self, batch, batch_idx):
        """Compute and return the training loss."""
        x, u, y = batch['x'], batch.get('u'), batch['h_one_hot'] 
        outputs = self.forward(x, y, u)
        loss = outputs['loss']
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Compute and log the validation loss."""
        x, y = batch['x'], batch['h_one_hot']
        outputs = self.forward(x, y)
        loss = outputs['loss']
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Perform the prediction and return logits and embeddings."""
        x, y = batch['x'], batch['h_one_hot']
        outputs = self.forward(x, y)

        h, _ = self.sage(outputs['x_cls'], self.edge_index.to(outputs['x_cls'].device))

        outputs = {
            'logits': outputs['logits'][:, :self.num_leaves],
            'labels': y[:, :self.num_leaves],
            'labels_h': y,
            'logits_h': outputs['logits_h'],
            'embeddings': self.backbone(x)[0],
            'h': h
            
        }

        return outputs

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Gather all model parameters to include in the optimizer
        parameters = list(self.backbone.parameters()) + list(self.fc.parameters())
        parameters += list(self.sage.parameters())

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