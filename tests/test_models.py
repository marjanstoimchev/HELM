"""
Tests for model architecture — HierarchicalVIT, layers, forward passes.
Requires timm + torch.
"""
import torch
import pytest

from tests.conftest import requires_timm, requires_pyg, requires_full_stack


# ═══════════════════════════════════════════════════════════════════════════
# 1.  HierarchicalVIT backbone
# ═══════════════════════════════════════════════════════════════════════════

@requires_timm
class TestHierarchicalVIT:
    def _make_model(self, num_classes=17):
        from models.model import h_deit_base_embedding
        return h_deit_base_embedding(num_classes=num_classes, pretrained=False)

    def test_instantiation(self):
        model = self._make_model(17)
        assert model is not None

    def test_num_classes_attribute(self):
        model = self._make_model(17)
        assert model.num_classes == 17

    def test_embed_dim(self):
        model = self._make_model(17)
        assert model.embed_dim == 768

    def test_forward_output_shapes(self):
        model = self._make_model(10)
        model.eval()
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            x_cls, patch_emb = model(x)
        # x_cls: (batch, num_classes, embed_dim)
        assert x_cls.shape == (2, 10, 768)
        # patch_emb: (batch, embed_dim) — mean-pooled
        assert patch_emb.shape == (2, 768)

    def test_different_num_classes(self):
        for nc in [5, 17, 28, 60]:
            model = self._make_model(nc)
            model.eval()
            x = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                x_cls, _ = model(x)
            assert x_cls.shape[1] == nc

    def test_cls_tokens_parameter(self):
        model = self._make_model(17)
        assert model.cls_tokens.shape == (1, 17, 768)

    def test_pos_embed_shapes(self):
        model = self._make_model(17)
        assert model.pos_embed_cls.shape == (1, 17, 768)
        # 224/16 = 14, 14*14 = 196 patches
        assert model.pos_embed_pat.shape == (1, 196, 768)

    def test_gradients_flow(self):
        model = self._make_model(5)
        x = torch.randn(1, 3, 224, 224)
        x_cls, patch_emb = model(x)
        loss = x_cls.sum() + patch_emb.sum()
        loss.backward()
        assert model.cls_tokens.grad is not None

    def test_parameter_count_reasonable(self):
        model = self._make_model(17)
        total = sum(p.numel() for p in model.parameters())
        # ViT-Base has ~86M params; with extra cls tokens it should be around that
        assert 80e6 < total < 100e6


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Classifier layer
# ═══════════════════════════════════════════════════════════════════════════

@requires_timm
class TestClassifierLayer:
    def test_forward(self):
        from models.layers import Classifier
        clf = Classifier(hidden_size=768, n_classes=17)
        x = torch.randn(4, 768)
        out = clf(x)
        assert out.shape == (4, 17)

    def test_gradients(self):
        from models.layers import Classifier
        clf = Classifier(hidden_size=768, n_classes=10)
        x = torch.randn(2, 768, requires_grad=True)
        out = clf(x)
        out.sum().backward()
        assert x.grad is not None


# ═══════════════════════════════════════════════════════════════════════════
# 3.  SAGE graph layer
# ═══════════════════════════════════════════════════════════════════════════

@requires_pyg
@requires_timm
class TestSAGELayer:
    def test_forward(self):
        from models.layers import SAGE
        sage = SAGE(dim_in=768, dim_h=64, in_classes=10, out_classes=10)
        # x: (batch, num_nodes, dim_in) — need to handle per-graph
        # SAGE operates on 2D node features + edge_index
        x = torch.randn(2, 10, 768)  # batch of 2 graphs, 10 nodes each
        edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
        h, logits = sage(x, edge_index)
        assert logits.shape == (2, 10)

    def test_output_shapes(self):
        from models.layers import SAGE
        sage = SAGE(dim_in=768, dim_h=64, in_classes=20, out_classes=20)
        x = torch.randn(4, 20, 768)
        edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        h, logits = sage(x, edge_index)
        assert h.shape[0] == 4
        assert logits.shape == (4, 20)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  GAT layer
# ═══════════════════════════════════════════════════════════════════════════

@requires_pyg
@requires_timm
class TestGATLayer:
    def test_forward(self):
        from models.layers import GAT
        gat = GAT(dim_in=768, dim_h=64, in_classes=10, out_classes=10)
        x = torch.randn(2, 10, 768)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
        logits = gat(x, edge_index)
        assert logits.shape == (2, 10)


# ═══════════════════════════════════════════════════════════════════════════
# 5.  PatchEmbed layer
# ═══════════════════════════════════════════════════════════════════════════

class TestPatchEmbed:
    def test_forward(self):
        from models.layers import PatchEmbed
        pe = PatchEmbed(img_size=224, patch_size=16, in_chans=3, embed_dim=768)
        x = torch.randn(2, 3, 224, 224)
        out = pe(x)
        assert out.shape == (2, 196, 768)  # 14*14=196 patches

    def test_different_patch_size(self):
        from models.layers import PatchEmbed
        pe = PatchEmbed(img_size=224, patch_size=32, in_chans=3, embed_dim=768)
        x = torch.randn(2, 3, 224, 224)
        out = pe(x)
        assert out.shape == (2, 49, 768)  # 7*7=49 patches
