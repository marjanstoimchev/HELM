import copy
import torch.nn as nn
import torch.nn.functional as F
from lightly.models.utils import deactivate_requires_grad
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead

class BYOL(nn.Module):
    def __init__(self, backbone, hidden_dim = 768):
        super().__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(hidden_dim, 1024, 256)
        self.prediction_head = BYOLPredictionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        _, y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p
    
    def forward_momentum(self, x):
        _, y = self.backbone_momentum.forward_features(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z
