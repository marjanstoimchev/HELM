"""
HELM (Hierarchical Embedding Learning Model) Lightning Module.

This module implements a unified training system with configurable loss components:
- Classification loss (BCE, Focal, ASL)
- Graph-based loss (for hierarchical structure)
- BYOL loss (for semi-supervised learning)

All loss components are configurable via YAML configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from typing import Dict, Any, Optional
from torchmetrics import F1Score, Precision, Recall, AveragePrecision

from ..models import h_deit_base_embedding, Classifier, SAGE, GCN, GAT
from ..models import BYOLProjectionHead, BYOLPredictionHead
from ..data.augmentations import create_byol_augmentations


class HELMModule(L.LightningModule):
    """
    Unified PyTorch Lightning module for hierarchical multi-label classification.

    Supports multiple training modes via configuration:
    - Supervised only (classification loss)
    - Graph-based (classification + graph loss)
    - BYOL SSL (classification + BYOL loss)
    - Full SSL (classification + graph + BYOL loss)
    """

    def __init__(
        self,
        num_classes: int,
        num_leaves: int,
        edge_index: Optional[torch.Tensor],
        leaf_indices: Optional[torch.Tensor],
        config: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['edge_index', 'leaf_indices'])

        self.num_classes = num_classes
        self.num_leaves = num_leaves
        self.config = config

        # Store leaf indices for extracting leaf predictions from full hierarchy
        self.register_buffer('leaf_indices', leaf_indices if leaf_indices is not None
                            else torch.arange(num_leaves))

        # Extract configuration
        model_cfg = config['model']
        train_cfg = config['training']
        loss_cfg = train_cfg['loss']

        # Store loss configuration
        self.use_classification_loss = loss_cfg['use_classification_loss']
        self.use_graph_loss = loss_cfg['use_graph_loss']
        self.use_byol_loss = loss_cfg['use_byol_loss']

        self.classification_weight = loss_cfg['classification_weight']
        self.graph_weight = loss_cfg['graph_weight']
        self.byol_weight = loss_cfg['byol_weight']

        # Initialize backbone
        self.backbone = self._create_backbone(model_cfg)

        # Initialize classifier head
        embed_dim = model_cfg['embed_dim']
        self.classifier = Classifier(embed_dim, num_classes)

        # Initialize graph module if needed
        if self.use_graph_loss and edge_index is not None:
            self.edge_index = edge_index
            self.graph_module = self._create_graph_module(model_cfg, embed_dim, num_classes)
        else:
            self.edge_index = None
            self.graph_module = None

        # Initialize BYOL components if needed
        if self.use_byol_loss:
            self.byol_projection = BYOLProjectionHead(
                embed_dim,
                model_cfg['byol_hidden_dim'],
                model_cfg['byol_projection_dim']
            )
            self.byol_prediction = BYOLPredictionHead(
                model_cfg['byol_projection_dim'],
                model_cfg['byol_hidden_dim'],
                model_cfg['byol_projection_dim']
            )

            # Momentum encoder
            self.momentum_backbone = self._create_backbone(model_cfg)
            self.momentum_projection = BYOLProjectionHead(
                embed_dim,
                model_cfg['byol_hidden_dim'],
                model_cfg['byol_projection_dim']
            )

            # Initialize momentum encoder with same weights
            self.momentum_backbone.load_state_dict(self.backbone.state_dict())
            self.momentum_projection.load_state_dict(self.byol_projection.state_dict())

            # Freeze momentum encoder
            for param in self.momentum_backbone.parameters():
                param.requires_grad = False
            for param in self.momentum_projection.parameters():
                param.requires_grad = False

            self.byol_momentum = model_cfg['byol_momentum']

            # Create augmentation pipelines for BYOL (weak and strong)
            image_size = config['dataset']['image_size']
            self.train_transforms_w, self.train_transforms_s = create_byol_augmentations(image_size)
        else:
            self.byol_projection = None
            self.byol_prediction = None
            self.momentum_backbone = None
            self.momentum_projection = None
            self.train_transforms_w = None
            self.train_transforms_s = None

        # Initialize loss functions
        self.classification_loss_fn = self._get_loss_fn(loss_cfg['classification_loss_type'])
        if self.use_graph_loss:
            self.graph_loss_fn = self._get_loss_fn(loss_cfg['graph_loss_type'])

        # Initialize metrics
        self._setup_metrics()

    def _create_backbone(self, model_cfg):
        """Create the hierarchical ViT backbone."""
        pretrained_path = model_cfg.get('pretrained_path')
        backbone = h_deit_base_embedding(
            num_classes=self.num_classes,
            pretrained=(pretrained_path is not None),
        )
        return backbone

    def _create_graph_module(self, model_cfg, embed_dim, num_classes):
        """Create the graph neural network module."""
        graph_type = model_cfg['graph_type']
        hidden_dim = model_cfg['graph_hidden_dim']
        dropout = model_cfg['graph_dropout']

        if graph_type == 'sage':
            return SAGE(embed_dim, hidden_dim, num_classes, num_classes)
        elif graph_type == 'gcn':
            return GCN(embed_dim, hidden_dim, num_classes)
        elif graph_type == 'gat':
            return GAT(embed_dim, hidden_dim, num_classes, num_classes)
        else:
            raise ValueError(f"Unknown graph type: {graph_type}")

    def _get_loss_fn(self, loss_type: str):
        """Get loss function by type."""
        if loss_type == 'bce':
            return nn.BCEWithLogitsLoss()
        elif loss_type == 'focal':
            # Simple focal loss implementation
            return nn.BCEWithLogitsLoss()  # TODO: Implement proper focal loss
        elif loss_type == 'asl':
            # ASL loss
            return nn.BCEWithLogitsLoss()  # TODO: Implement ASL
        else:
            return nn.BCEWithLogitsLoss()

    def _setup_metrics(self):
        """Initialize evaluation metrics."""
        self.train_f1 = F1Score(task='multilabel', num_labels=self.num_leaves, average='macro')
        self.val_f1 = F1Score(task='multilabel', num_labels=self.num_leaves, average='macro')
        self.test_f1 = F1Score(task='multilabel', num_labels=self.num_leaves, average='macro')

        self.val_precision = Precision(task='multilabel', num_labels=self.num_leaves, average='macro')
        self.val_recall = Recall(task='multilabel', num_labels=self.num_leaves, average='macro')

    def _prepare_data(self, x, u=None):
        """
        Prepare data with appropriate augmentations based on presence of unlabeled data.

        This method implements the correct BYOL data preparation strategy:
        - If u is None (supervised mode with BYOL): Apply weak and strong augmentations to x
        - If u is not None (semi-supervised mode): Concatenate labeled and unlabeled, then apply augmentations
        - If no BYOL: Just concatenate if u exists, or return x

        Args:
            x: Labeled images [n_labeled, C, H, W]
            u: Unlabeled images [n_unlabeled, C, H, W] or None

        Returns:
            Tuple of (view1, view2) where:
            - view1: First augmented view (weak)
            - view2: Second augmented view (strong) or None if no BYOL
        """
        # Case 1: Supervised with BYOL (u is None but BYOL enabled)
        if self.use_byol_loss and u is None:
            # Apply weak and strong augmentations to labeled data
            x0 = self.train_transforms_w(x)  # Weak augmentation
            x1 = self.train_transforms_s(x)  # Strong augmentation
            return x0, x1

        # Case 2: Semi-supervised mode (u is not None)
        if u is not None:
            # Concatenate labeled and unlabeled into single batch
            x_combined = torch.cat([x, u], dim=0)  # [n_labeled + n_unlabeled, C, H, W]

            if self.use_byol_loss:
                # Apply weak and strong augmentations to combined batch
                x0 = self.train_transforms_w(x_combined)  # Weak aug to all
                x1 = self.train_transforms_s(x_combined)  # Strong aug to all
                return x0, x1
            else:
                # No BYOL: just return combined batch
                return x_combined, None

        # Case 3: Supervised without BYOL (u is None, BYOL disabled)
        return x, None

    def _compute_classification_loss(self, x_cls, y):
        """Compute classification loss."""
        # Average over class tokens
        embeddings = x_cls.mean(dim=1)  # [batch_size, embed_dim]
        logits = self.classifier(embeddings)  # [batch_size, num_classes]

        loss = self.classification_loss_fn(logits[:len(y)], y)
        return logits, loss

    def _compute_graph_loss(self, x_cls, y):
        """Compute graph-based loss using GNN."""
        if self.graph_module is None or self.edge_index is None:
            return None, 0.0

        _, logits = self.graph_module(x_cls, self.edge_index.to(x_cls.device))
        loss = self.graph_loss_fn(logits[:len(y)], y)
        return logits, loss

    def _compute_byol_loss(self, x1, x2, embeddings_x1, embeddings_x2):
        """Compute BYOL loss."""
        # Online network
        z1 = self.byol_projection(embeddings_x1)
        p1 = self.byol_prediction(z1)

        z2 = self.byol_projection(embeddings_x2)
        p2 = self.byol_prediction(z2)

        # Target network (momentum encoder)
        with torch.no_grad():
            if self.momentum_backbone is not None:
                _, target_embeddings_x1 = self.momentum_backbone(x1)
                _, target_embeddings_x2 = self.momentum_backbone(x2)

                target_z1 = self.momentum_projection(target_embeddings_x1)
                target_z2 = self.momentum_projection(target_embeddings_x2)

        # Compute loss (negative cosine similarity)
        loss1 = 2 - 2 * F.cosine_similarity(p1, target_z2.detach(), dim=-1).mean()
        loss2 = 2 - 2 * F.cosine_similarity(p2, target_z1.detach(), dim=-1).mean()

        return (loss1 + loss2) / 2

    def forward(self, x, y, u=None):
        """
        Forward pass with configurable loss computation.

        Args:
            x: Input images (labeled)
            y: Labels (hierarchical one-hot)
            u: Unlabeled images (optional, for BYOL)

        Returns:
            Dictionary with logits and loss components
        """
        x_view1, x_view2 = self._prepare_data(x, u)

        # Forward through backbone
        x_cls, embeddings = self.backbone(x_view1)

        outputs = {}
        total_loss = 0.0

        # Classification loss
        if self.use_classification_loss:
            logits_cls, loss_cls = self._compute_classification_loss(x_cls, y)
            outputs['logits_cls'] = logits_cls
            outputs['loss_cls'] = loss_cls
            total_loss += self.classification_weight * loss_cls

        # Graph loss
        if self.use_graph_loss:
            logits_graph, loss_graph = self._compute_graph_loss(x_cls, y)
            if logits_graph is not None:
                outputs['logits_graph'] = logits_graph
                outputs['loss_graph'] = loss_graph
                total_loss += self.graph_weight * loss_graph

        # BYOL loss
        if self.use_byol_loss and x_view2 is not None:
            _, embeddings_x2 = self.backbone(x_view2)
            loss_byol = self._compute_byol_loss(x_view1, x_view2, embeddings, embeddings_x2)
            outputs['loss_byol'] = loss_byol
            total_loss += self.byol_weight * loss_byol

        # Combine logits for final prediction
        final_logits = outputs.get('logits_cls', None)
        if self.use_graph_loss and 'logits_graph' in outputs:
            if final_logits is not None:
                final_logits = final_logits + outputs['logits_graph']
            else:
                final_logits = outputs['logits_graph']

        # Extract leaf predictions using leaf_indices
        outputs['logits'] = final_logits[:, self.leaf_indices] if final_logits is not None else None
        outputs['logits_h'] = final_logits
        outputs['loss'] = total_loss

        return outputs

    def training_step(self, batch, batch_idx):
        """Training step."""
        x = batch['images']
        y = batch['labels']
        u = batch.get('u', None)

        outputs = self(x, y, u)
        loss = outputs['loss']

        # Log losses
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        if 'loss_cls' in outputs:
            self.log('train_loss_cls', outputs['loss_cls'], on_step=False, on_epoch=True)
        if 'loss_graph' in outputs:
            self.log('train_loss_graph', outputs['loss_graph'], on_step=False, on_epoch=True)
        if 'loss_byol' in outputs:
            self.log('train_loss_byol', outputs['loss_byol'], on_step=False, on_epoch=True)

        # Log metrics (only on labeled portion)
        if outputs['logits'] is not None:
            preds = torch.sigmoid(outputs['logits'])
            leaf_labels = batch['leaf_labels']
            # IMPORTANT: Only compute metrics on labeled samples (first len(y) samples)
            preds_labeled = preds[:len(y)]
            self.train_f1(preds_labeled, leaf_labels.int())
            self.log('train_f1', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        # Update momentum encoder
        if self.use_byol_loss:
            self._update_momentum_encoder()

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x = batch['images']
        y = batch['labels']

        outputs = self(x, y, None)
        loss = outputs['loss']

        # Log losses
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log metrics
        if outputs['logits'] is not None:
            preds = torch.sigmoid(outputs['logits'])
            leaf_labels = batch['leaf_labels']

            self.val_f1(preds, leaf_labels.int())
            self.val_precision(preds, leaf_labels.int())
            self.val_recall(preds, leaf_labels.int())

            self.log('val_f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
            self.log('val_f1_macro', self.val_f1, on_step=False, on_epoch=True)
            self.log('val_precision', self.val_precision, on_step=False, on_epoch=True)
            self.log('val_recall', self.val_recall, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        x = batch['images']
        y = batch['labels']

        outputs = self(x, y, None)

        # Log metrics
        if outputs['logits'] is not None:
            preds = torch.sigmoid(outputs['logits'])
            leaf_labels = batch['leaf_labels']

            self.test_f1(preds, leaf_labels.int())
            self.log('test_f1', self.test_f1, on_step=False, on_epoch=True)

        return outputs

    def _update_momentum_encoder(self):
        """Update momentum encoder using EMA."""
        if self.momentum_backbone is None:
            return

        for param_online, param_momentum in zip(
            self.backbone.parameters(), self.momentum_backbone.parameters()
        ):
            param_momentum.data = (
                self.byol_momentum * param_momentum.data +
                (1 - self.byol_momentum) * param_online.data
            )

        for param_online, param_momentum in zip(
            self.byol_projection.parameters(), self.momentum_projection.parameters()
        ):
            param_momentum.data = (
                self.byol_momentum * param_momentum.data +
                (1 - self.byol_momentum) * param_online.data
            )

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        train_cfg = self.config['training']

        # Create parameter groups
        params = [
            {
                'params': self.backbone.parameters(),
                'lr': train_cfg['lr'],
            },
            {
                'params': self.classifier.parameters(),
                'lr': train_cfg['head_lr'],
            },
        ]

        if self.graph_module is not None:
            params.append({
                'params': self.graph_module.parameters(),
                'lr': train_cfg['head_lr'],
            })

        if self.use_byol_loss:
            params.extend([
                {'params': self.byol_projection.parameters(), 'lr': train_cfg['lr']},
                {'params': self.byol_prediction.parameters(), 'lr': train_cfg['lr']},
            ])

        # Create optimizer
        optimizer_type = train_cfg['optimizer'].lower()
        if optimizer_type == 'adamw':
            optimizer = torch.optim.AdamW(params, weight_decay=train_cfg['weight_decay'])
        elif optimizer_type == 'adam':
            optimizer = torch.optim.Adam(params, weight_decay=train_cfg['weight_decay'])
        elif optimizer_type == 'sgd':
            optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=train_cfg['weight_decay'])
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")

        # Create scheduler if needed
        if not train_cfg['use_scheduler']:
            return optimizer

        scheduler_type = train_cfg['scheduler_type']
        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=train_cfg['max_epochs'],
                eta_min=train_cfg['min_lr']
            )
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss'
                }
            }
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")

        return [optimizer], [scheduler]
