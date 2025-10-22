import torch
import torch.nn as nn
import math
from functools import partial
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.layers import trunc_normal_, to_2tuple
from .layers import SAGE






class HierarchicalVIT(VisionTransformer):
    def __init__(self, num_classes, num_heads, input_size=224, *args, **kwargs):   
        self.num_heads = num_heads

        super().__init__(*args, **kwargs)

        img_size = to_2tuple(input_size)
        patch_size = to_2tuple(self.patch_embed.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.num_patches = num_patches
 
        self.num_classes = num_classes
        self.cls_tokens = nn.Parameter(torch.zeros(1, num_classes, self.embed_dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, num_classes, self.embed_dim))
        self.pos_embed_pat = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))

        nn.init.trunc_normal_(self.cls_tokens, std=.02)
        nn.init.trunc_normal_(self.pos_embed_cls, std=.02)
        nn.init.trunc_normal_(self.pos_embed_pat, std=.02)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - self.num_classes
        N = self.num_patches
        if npatch == N and w == h:
            return self.pos_embed_pat
        patch_pos_embed = self.pos_embed_pat
        dim = x.shape[-1]

        w0 = w // self.patch_embed.patch_size[0]
        h0 = h // self.patch_embed.patch_size[0]

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return patch_pos_embed

    def forward_features(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        pos_embed_pat = self.interpolate_pos_encoding(x, w, h)
        x = x + pos_embed_pat

        cls_tokens = self.cls_tokens.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.pos_embed_cls

        x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)

        x_cls = x[:, :self.num_classes]
        return x_cls, x[:, self.num_classes:]

    def forward(self, x, y=None):
        x_cls, patch_embeddings = self.forward_features(x)
        return x_cls, patch_embeddings.mean(1)

def h_deit_base(num_classes, num_leaves, pretrained=False, edge_index = None, **kwargs):
    model = HierarchicalVIT(
        num_classes=num_classes,  
        num_leaves=num_leaves, 
        num_heads=12,  # Ensure the number of heads matches the number of hierarchical levels
        edge_index=edge_index,
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    model.default_cfg = _cfg()

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            # url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
            map_location="cpu", check_hash=True
        )#['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['cls_token', 'pos_embed']}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Pre-trained weights loaded successfully.")
    return model

def h_deit_base_embedding(num_classes, pretrained=False, **kwargs):
    model = HierarchicalVIT(
        num_classes=num_classes,  
        num_heads=12,  # Ensure the number of heads matches the number of hierarchical levels
        patch_size=16, 
        embed_dim=768, 
        depth=12, 
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs
    )
    model.default_cfg = _cfg()

    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            # url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
            map_location="cpu", check_hash=True
        )#['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['cls_token', 'pos_embed']}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print("Pre-trained weights loaded successfully.")
    return model