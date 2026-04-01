import copy
import torch
import torch.nn as nn
import lightning as L

from utils.utils import Dotdict
from augmentations import DataAugmentation

# from models.losses import HierarchicalContrastiveLoss, HierarchicalTripletLoss
from lightly.utils.scheduler import cosine_schedule
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum

from models.losses import CosineSimilarity
from models.layers import Classifier

from utils.utils import Dotdict
from augmentations import DataAugmentation
from models.layers import SAGE, Classifier

class SemiSupervisedGraphBYOLModel(L.LightningModule):
    """
    A LightningModule for semi-supervised learning with BYOL.
    """

    def __init__(self, config, backbone, num_leaves, learning_task='hmlc', edge_index=None):
        """
        Initialize the semi-supervised model with BYOL.
        Args:
            config (dict): Configuration parameters.
            backbone (nn.Module): Backbone neural network model.
            num_leaves (int): Number of output nodes in the final classifier layer.
        """
        super().__init__()
        self.backbone = backbone
        self.fc = Classifier(backbone.num_classes, backbone.num_classes)
        self.sage = SAGE(backbone.embed_dim, 64, backbone.num_classes, backbone.num_classes)
        self.edge_index = edge_index
        self.num_leaves = num_leaves

        self.projection_head = BYOLProjectionHead(backbone.embed_dim, 2048, 256)
        self.prediction_head = BYOLPredictionHead(256, 2048, 256)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

        self.loss_cls = nn.BCEWithLogitsLoss()
        self.loss_gcn = nn.BCEWithLogitsLoss()
        self.loss_byol = CosineSimilarity()

        self.train_transforms_w = DataAugmentation(mode='weak')
        self.train_transforms_s = DataAugmentation(mode='strong')

        # Save hyperparameters
        training_params = config.training.to_dict() if isinstance(config.training, Dotdict) else config.training
        dataset_params = config.dataset.to_dict() if isinstance(config.dataset, Dotdict) else config.dataset
        hparams = {**training_params, **dataset_params}
        self.save_hyperparameters(hparams)

    def _prepare_data(self, x, u=None):
        """Prepare data with appropriate augmentations based on the presence of graph data and the mode."""
        x0_l = self.train_transforms_w(x)  # Weak augmentation for labeled data
        
        if hasattr(self, 'projection_head') and u is None:   # supervised with BYOL
            x1_l = self.train_transforms_s(x)  # Strong augmentation for labeled data
            return x0_l, x1_l
        
        if u is not None:  # Semi-supervised mode with unlabeled data
            x0_u = self.train_transforms_w(u)
            x0 = torch.cat((x0_l, x0_u), dim=0)  # Combine weak augmentations
            
            if hasattr(self, 'projection_head'):  # Semi-supervised with BYOL
                x1_l = self.train_transforms_s(x)  # Strong augmentation for labeled data
                x1_u = self.train_transforms_s(u)
                x1 = torch.cat((x1_l, x1_u), dim=0)  # Combine strong augmentations
                return x0, x1
            
            return x0, None  # Semi-supervised without BYOL

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
    
    def _compute_byol_loss(self, x0, x1, embeddings_x0, embeddings_x1):
        """Compute BYOL loss for semi-supervised learning."""
        momentum = cosine_schedule(self.current_epoch, 100, 0.996, 1)
        update_momentum(self.backbone, self.backbone_momentum, m=momentum)
        update_momentum(self.projection_head, self.projection_head_momentum, m=momentum)

        p0 = self._project_byol(embeddings_x0)
        z0 = self._project_momentum(x0)
        p1 = self._project_byol(embeddings_x1)
        z1 = self._project_momentum(x1)

        return self.loss_byol(p0, z1) + self.loss_byol(p1, z0)

    def _project_byol(self, embeddings):
        """Project data using the BYOL projection and prediction heads."""
        return self.prediction_head(self.projection_head(embeddings))

    def _project_momentum(self, embeddings):
        """Project data using the momentum backbone and projection head."""
        _, y = self.backbone_momentum(embeddings)
        return self.projection_head_momentum(y).detach()
    
    def forward(self, x, y, u=None):
        """Forward pass through the backbone and classifier."""
        x0, x1 = self._prepare_data(x, u)  # Prepare data with appropriate augmentations

        x_cls, embeddings_x0 = self.backbone(x0)
        _, embeddings_x1 = self.backbone(x1)

        # Compute classifier logits and loss
        logits_cls, loss_cls = self._compute_cls_loss(x_cls, y)  

        # Compute BYOL loss if in semi-supervised mode with BYOL
        loss_byol = self._compute_byol_loss(x0, x1, embeddings_x0, embeddings_x1) if x1 is not None else 0

        logits_gcn, loss_gcn = self._compute_gcn_loss(x_cls, y) 

        # Total loss is the sum of classifier, GCN, and BYOL losses
        total_loss = loss_cls + loss_byol + loss_gcn

        return {
            'logits_h': logits_cls + logits_gcn,
            'logits': logits_cls[:, :self.num_leaves] + logits_gcn[:, :self.num_leaves],
            'loss': total_loss,  # Total loss
            'x_cls': x_cls,
        }

    def training_step(self, batch, batch_idx):
        """Compute and return the training loss."""
        x, u, y = batch['x'], batch.get('u'), batch['h_one_hot'] 
        outputs = self.forward(x, y, u)
        loss = outputs['loss']
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def _forward_eval(self, x, y):
        """Forward pass for validation/prediction — no BYOL, no augmentation."""
        x_cls, _ = self.backbone(x)
        logits_cls, loss_cls = self._compute_cls_loss(x_cls, y)
        logits_gcn, loss_gcn = self._compute_gcn_loss(x_cls, y)

        return {
            'logits_h': logits_cls + logits_gcn,
            'logits': logits_cls[:, :self.num_leaves] + logits_gcn[:, :self.num_leaves],
            'loss': loss_cls + loss_gcn,
            'x_cls': x_cls,
        }

    def validation_step(self, batch, batch_idx):
        """Compute and log the validation loss (cls + graph only, no BYOL)."""
        x, y = batch['x'], batch['h_one_hot']
        loss = self._forward_eval(x, y)['loss']
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Prediction — single backbone pass, no BYOL, no augmentation."""
        x, y = batch['x'], batch['h_one_hot']
        outputs = self._forward_eval(x, y)

        h, _ = self.sage(outputs['x_cls'], self.edge_index.to(outputs['x_cls'].device))

        return {
            'logits': outputs['logits'][:, :self.num_leaves],
            'labels': y[:, :self.num_leaves],
            'labels_h': y,
            'logits_h': outputs['logits_h'],
            'embeddings': outputs['x_cls'],
            'h': h,
        }

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        parameters = list(self.backbone.parameters()) + list(self.fc.parameters())
        parameters += list(self.projection_head.parameters()) + list(self.prediction_head.parameters())
        parameters += list(self.backbone_momentum.parameters()) + list(self.projection_head_momentum.parameters())
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