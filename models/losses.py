import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity
import torch.nn.functional as F

class ZLPRLoss(nn.Module):
    def __init__(self):
        super(ZLPRLoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Mask out padding tokens (assumed to be -100)
        loss_mask = y_true != -100
        y_true = y_true[loss_mask].view(-1, y_pred.size(-1))
        y_pred = y_pred[loss_mask].view(-1, y_true.size(-1))
        
        # Custom loss computation
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[:, :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
        
        return (neg_loss + pos_loss).mean()
    
class CosineSimilarity(torch.nn.Module):
    """Implementation of the Negative Cosine Simililarity used in the SimSiam[0] paper.

    [0] SimSiam, 2020, https://arxiv.org/abs/2011.10566

    Examples:

        >>> # initialize loss function
        >>> loss_fn = NegativeCosineSimilarity()
        >>>
        >>> # generate two representation tensors
        >>> #Ã‚Â with batch size 10 and dimension 128
        >>> x0 = torch.randn(10, 128)
        >>> x1 = torch.randn(10, 128)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(x0, x1)
    """

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        """Same parameters as in torch.nn.CosineSimilarity

        Args:
            dim (int, optional):
                Dimension where cosine similarity is computed. Default: 1
            eps (float, optional):
                Small value to avoid division by zero. Default: 1e-8
        """
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        return 2-2*cosine_similarity(x0, x1, self.dim, self.eps).mean()


class HierarchicalContrastiveLoss(nn.Module):
    def __init__(self, hierarchy, temperature=0.07):
        super(HierarchicalContrastiveLoss, self).__init__()
        self.hierarchy = hierarchy
        self.temperature = temperature

        # Create a mapping from each label to its ancestors
        self.label_to_ancestors = {}
        for leaf, ancestors in hierarchy.items():
            for ancestor in ancestors:
                if ancestor not in self.label_to_ancestors:
                    self.label_to_ancestors[ancestor] = set()
                self.label_to_ancestors[ancestor].add(leaf)

        
    
    def cosine_distance(self, x1, x2):
        x1 = F.normalize(x1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        return 1 - torch.matmul(x1, x2.T)

    def forward(self, features, target):
        """
        :param features: shape (batch_size, num_labels, embed_dim)
        :param target: shape (batch_size, num_labels)
        """
        device = features.device
        batch_size, num_labels, embed_dim = features.shape

        # Compute pairwise cosine distances
        feature_matrix = features.contiguous().view(batch_size * num_labels, embed_dim)
        distance_matrix = self.cosine_distance(feature_matrix, feature_matrix)  # shape: (batch_size*num_labels, batch_size*num_labels)

        # Create a mask for positive pairs (same hierarchy path) and negative pairs (different hierarchy levels)
        positive_mask = torch.zeros((batch_size * num_labels, batch_size * num_labels), dtype=torch.float32).to(device)
        negative_mask = torch.ones((batch_size * num_labels, batch_size * num_labels), dtype=torch.float32).to(device)

        for i in range(batch_size):
            for leaf, ancestors in self.hierarchy.items():
                for ancestor in ancestors:
                    if target[i, ancestor] == 1:
                        for other_leaf in self.label_to_ancestors[ancestor]:
                            pos_idx = i * num_labels + leaf
                            anc_idx = i * num_labels + other_leaf
                            positive_mask[pos_idx, anc_idx] = 1
                            negative_mask[pos_idx, anc_idx] = 0

        # Compute the positive and negative parts of the loss
        positive_distances = positive_mask * distance_matrix
        negative_distances = negative_mask * distance_matrix

        pos_loss = torch.sum(positive_distances) / torch.sum(positive_mask)
        neg_loss = torch.sum(torch.exp(-negative_distances / self.temperature)) / torch.sum(negative_mask)

        # Adjusted to ensure non-negative loss
        loss = pos_loss + torch.max(torch.tensor(0.0).to(device), torch.log(neg_loss + 1e-6))

        return loss
    
class HierarchicalTripletLoss(nn.Module):
    def __init__(self, hierarchy, margin=1.0, distance_metric='cosine', device='cuda'):
        super(HierarchicalTripletLoss, self).__init__()
        self.hierarchy = hierarchy
        self.margin = margin
        self.distance_metric = distance_metric
        self.device = device
        
        # Create ancestor-leaf mappings for quick lookup
        self.ancestor_to_leaves = self._create_ancestor_to_leaves_map(hierarchy)
    
    def _create_ancestor_to_leaves_map(self, hierarchy):
        ancestor_to_leaves = {}
        for leaf, ancestors in hierarchy.items():
            for ancestor in ancestors:
                if ancestor not in ancestor_to_leaves:
                    ancestor_to_leaves[ancestor] = set()
                ancestor_to_leaves[ancestor].add(leaf)
        return ancestor_to_leaves

    def _get_distance(self, x1, x2):
        if self.distance_metric == 'cosine':
            x1 = F.normalize(x1, p=2, dim=-1)
            x2 = F.normalize(x2, p=2, dim=-1)
            return 1 - torch.matmul(x1, x2.T)
        elif self.distance_metric == 'euclidean':
            return torch.cdist(x1, x2, p=2)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
    
    def _get_level_difference(self, anchor, negative, device):
        # Handle cases where labels might not be in the hierarchy
        anchor_levels = torch.tensor(
            [len(self.hierarchy.get(label, [])) for label in anchor], device=device)
        negative_levels = torch.tensor(
            [len(self.hierarchy.get(label, [])) for label in negative], device=device)
        level_diff = torch.abs(anchor_levels - negative_levels)
        return level_diff
    
    def forward(self, features, labels):
        """
        :param features: shape (batch_size, num_labels, embed_dim)
        :param labels: shape (batch_size, num_labels)
        """
        batch_size, num_labels, embed_dim = features.shape
        triplet_loss = 0.0
        num_triplets = 0
        device = features.device
        
        for i in range(batch_size):
            for anchor_label, ancestors in self.hierarchy.items():
                if labels[i, anchor_label] == 1:
                    anchor_embedding = features[i, anchor_label].unsqueeze(0)
                    
                    # Generate positive examples
                    for ancestor in ancestors:
                        if labels[i, ancestor] == 1:
                            positive_embedding = features[i, ancestor].unsqueeze(0)
                            
                            # Generate negative examples
                            for other_label in range(num_labels):
                                if labels[i, other_label] == 1 and other_label not in ancestors:
                                    negative_embedding = features[i, other_label].unsqueeze(0)
                                    level_diff = self._get_level_difference([anchor_label], [other_label], device)
                                    distance_ap = self._get_distance(anchor_embedding, positive_embedding)
                                    distance_an = self._get_distance(anchor_embedding, negative_embedding)
                                    
                                    loss = F.relu(distance_ap - distance_an + self.margin * level_diff)
                                    triplet_loss += loss
                                    num_triplets += 1
        
        return triplet_loss / num_triplets if num_triplets > 0 else triplet_loss

# Asymmetric Loss For Multi-Label Classification
# Query2Label: A Simple Transformer Way to Multi-Label Classification

class AsymmetricLoss(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-5, disable_torch_grad_focal_loss=False):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                with torch.no_grad():
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(False)
                    self.xs_pos = self.xs_pos * self.targets
                    self.xs_neg = self.xs_neg * self.anti_targets
                    self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                                self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
                    # if self.disable_torch_grad_focal_loss:
                    #     torch._C.set_grad_enabled(True)
                self.loss *= self.asymmetric_w
            else:
                self.xs_pos = self.xs_pos * self.targets
                self.xs_neg = self.xs_neg * self.anti_targets
                self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                            self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)   
                self.loss *= self.asymmetric_w         
        _loss = - self.loss.sum() / x.size(0)
        _loss = _loss / y.size(1) * 1000

        return _loss