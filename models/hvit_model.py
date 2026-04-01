import torch
import torch.nn as nn
from functools import partial
from timm.models.vision_transformer import _cfg
from collections import defaultdict
            
import math
from timm.models.layers import trunc_normal_, to_2tuple
from models.layers import Block, PatchEmbed

class HierarchicalViT(nn.Module):
    def __init__(self, num_classes, patch_size, embed_dim, mlp_ratio, qkv_bias, num_heads, img_size=224, in_chans=3, depth=12, norm_layer=nn.LayerNorm, *args, **kwargs):   
        

        super().__init__(*args, **kwargs)

        self.num_heads = num_heads
        img_size = to_2tuple(img_size)
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        patch_size = to_2tuple(self.patch_embed.patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.num_classes = num_classes
 
        self.cls_tokens = nn.Parameter(torch.zeros(1, num_classes, embed_dim))
        self.pos_embed_cls = nn.Parameter(torch.zeros(1, num_classes, embed_dim))
        self.pos_embed_pat = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer) 
             for _ in range(depth)]
             )
    
        self.norm_layer = norm_layer(embed_dim)
        
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

    def forward_features(self, x, n=12):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)
        pos_embed_pat = self.interpolate_pos_encoding(x, w, h)
        x = x + pos_embed_pat

        cls_tokens = self.cls_tokens.expand(B, -1, -1)
        cls_tokens = cls_tokens + self.pos_embed_cls

        x = torch.cat((cls_tokens, x), dim=1)

        # Transformer blocks
        attn_weights = []
        for i, blk in enumerate(self.blocks):
            x, weights = blk(x)
            if len(self.blocks) - i <= n:
                attn_weights.append(weights)

        x = self.norm_layer(x)
        x_cls = x[:, :self.num_classes]          # Classification tokens
        patch_embeddings = x[:, self.num_classes:]  # Patch embeddings

        return x_cls, patch_embeddings, attn_weights
    
    def forward(self, x, y=None):
        x_cls, patch_embeddings, _ = self.forward_features(x)
        return x_cls, patch_embeddings.mean(1)
              
def h_vit_base_patch16(num_classes, pretrained=False, **kwargs):
    model = HierarchicalViT(
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
        # Load the pretrained encoder weights
        checkpoint = torch.hub.load_state_dict_from_url(
            url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
            map_location="cpu", check_hash=True
        )
        
        model_dict = model.state_dict()  # Extract encoder's state_dict
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}  # Match only encoder layers
        
        # Filter out 'cls_token' and 'pos_embed' since we don't want to load them from pretrained
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in ['cls_token', 'pos_embed']}
        
        # Update the model's encoder state dict
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)  # Load pretrained encoder weights into encoder part
        
        print("Pre-trained encoder weights loaded successfully.")
    
    return model