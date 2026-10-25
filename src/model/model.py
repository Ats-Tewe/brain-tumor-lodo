
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, Tuple, Dict, List

class EfficientNetB0Backbone(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        full_model = timm.create_model(
            'efficientnet_b0', pretrained=pretrained,
            num_classes=0, global_pool='',
        )
        self.backbone = full_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

class PatchTokenizationBridge(nn.Module):
    def __init__(self, cnn_channels=1280, vit_embed_dim=384, num_patches=49, dropout=0.1):
        super().__init__()
        self.num_patches = num_patches
        self.vit_embed_dim = vit_embed_dim
        self.projection = nn.Linear(cnn_channels, vit_embed_dim, bias=True)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, vit_embed_dim))
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, vit_embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(vit_embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embeddings, std=0.02)
        nn.init.trunc_normal_(self.projection.weight, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        B = feature_map.size(0)
        tokens = feature_map.flatten(2).transpose(1, 2)
        tokens = self.projection(tokens)
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.position_embeddings
        tokens = self.dropout(tokens)
        tokens = self.norm(tokens)
        return tokens

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, attn_dropout=0.1, proj_dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_weights = attn.clone()
        attn = self.attn_drop(attn)
        output = (attn @ v).transpose(1, 2).reshape(B, N, D)
        output = self.proj(output)
        output = self.proj_drop(output)
        return output, attn_weights

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, mlp_ratio=4.0, attn_dropout=0.1, mlp_dropout=0.1):
        super().__init__()
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-6)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, attn_dropout)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden), nn.GELU(), nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, embed_dim), nn.Dropout(mlp_dropout),
        )

    def forward(self, x):
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights

class DualHeadHybridCNNViT(nn.Module):
    """
    Dual-Head Shared-Encoder Hybrid CNN-ViT for Brain Tumor Classification.

    v2.1 GRADIENT ISOLATION:
      During TRAINING, the CNN auxiliary head receives backbone features via
      stop-gradient (feature_map.detach()). This ensures the backbone is
      optimized exclusively for the ViT primary head. The CNN head learns
      to classify from features it cannot influence.

      During EVALUATION (including Grad-CAM), gradients flow normally through
      the CNN head to the backbone for visualization purposes only (no weight
      updates occur at eval time).

      This dual-level isolation:
        Training:  stop-gradient prevents CNN from shaping backbone features
        Inference: architectural separation prevents Grad-CAM gradients
                   from reaching ViT weights (verified in Module 2 checks)
    """

    def __init__(
        self,
        num_classes: int = 4,
        efficientnet_pretrained: bool = True,
        img_size: int = 224,
        vit_embed_dim: int = 384,
        vit_num_heads: int = 6,
        vit_num_layers: int = 12,
        vit_mlp_ratio: float = 4.0,
        bridge_dropout: float = 0.1,
        vit_attn_dropout: float = 0.1,
        vit_mlp_dropout: float = 0.1,
        cnn_aux_dropout: float = 0.3,
        mc_dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.vit_embed_dim = vit_embed_dim
        self.vit_num_layers = vit_num_layers
        self.mc_dropout_rate = mc_dropout_rate
        self.img_size = img_size

        spatial = img_size // 32
        num_patches = spatial * spatial
        self._spatial = spatial
        self._num_patches = num_patches

        self.backbone = EfficientNetB0Backbone(pretrained=efficientnet_pretrained)
        self._cnn_channels = 1280

        self.cnn_gap = nn.AdaptiveAvgPool2d(1)
        self.cnn_aux_dropout = nn.Dropout(p=cnn_aux_dropout)
        self.cnn_aux_fc = nn.Linear(self._cnn_channels, num_classes)

        self.bridge = PatchTokenizationBridge(
            cnn_channels=self._cnn_channels,
            vit_embed_dim=vit_embed_dim,
            num_patches=num_patches,
            dropout=bridge_dropout,
        )

        self.vit_blocks = nn.ModuleList([
            TransformerEncoderBlock(
                embed_dim=vit_embed_dim, num_heads=vit_num_heads,
                mlp_ratio=vit_mlp_ratio, attn_dropout=vit_attn_dropout,
                mlp_dropout=vit_mlp_dropout,
            ) for _ in range(vit_num_layers)
        ])
        self.vit_norm = nn.LayerNorm(vit_embed_dim, eps=1e-6)

        self.mc_dropout = nn.Dropout(p=mc_dropout_rate)
        self.vit_primary_fc = nn.Linear(vit_embed_dim, num_classes)

        self._init_head_weights()

    def _init_head_weights(self):
        for module in [self.cnn_aux_fc, self.vit_primary_fc]:
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
        for module in self.bridge.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, return_attention=False):
        feature_map = self.backbone(x)

        cnn_input = feature_map.detach() if self.training else feature_map
        cnn_pooled = self.cnn_gap(cnn_input).flatten(1)
        cnn_pooled = self.cnn_aux_dropout(cnn_pooled)
        cnn_logits = self.cnn_aux_fc(cnn_pooled)

        tokens = self.bridge(feature_map)
        attention_list = []
        for block in self.vit_blocks:
            tokens, attn_w = block(tokens)
            if return_attention:
                attention_list.append(attn_w)
        tokens = self.vit_norm(tokens)
        cls_token = tokens[:, 0, :]
        cls_token = self.mc_dropout(cls_token)
        vit_logits = self.vit_primary_fc(cls_token)

        output = {'vit_logits': vit_logits, 'cnn_logits': cnn_logits, 'feature_map': feature_map}
        if return_attention:
            output['attention_list'] = attention_list
        return output

    def enable_mc_dropout(self):
        self.eval()
        self.mc_dropout.train()

    def count_parameters(self):
        def count(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {
            'backbone_efficientnet_b0': count(self.backbone),
            'patch_tokenization_bridge': count(self.bridge),
            'vit_transformer_12_blocks': count(self.vit_blocks),
            'vit_layer_norm': count(self.vit_norm),
            'cnn_auxiliary_head': count(self.cnn_aux_fc),
            'vit_primary_head': count(self.vit_primary_fc),
            'total_trainable': count(self),
        }

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(-1)
        if self.label_smoothing > 0:
            with torch.no_grad():
                smooth_targets = torch.full_like(logits, self.label_smoothing / num_classes)
                smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing + self.label_smoothing / num_classes)

        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - pt) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            focal_weight = alpha_t * focal_weight

        if self.label_smoothing > 0:
            loss = -(focal_weight.unsqueeze(1) * smooth_targets * log_probs).sum(dim=-1)
        else:
            loss = -(focal_weight * log_pt)

        return loss.mean() if self.reduction == 'mean' else loss.sum()

def compute_class_weights(class_counts, class_order=None):
    if class_order is None:
        class_order = ['glioma', 'meningioma', 'pituitary', 'notumor']
    counts = [class_counts[c] for c in class_order]
    total = sum(counts)
    num_classes = len(counts)
    weights = [total / (num_classes * c) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)

def build_model(num_classes=4, pretrained=True, mc_dropout_rate=0.5, img_size=224):
    """
    Factory function: builds DualHeadHybridCNNViT.
    v2.1: CNN auxiliary head uses stop-gradient during training.
    """
    return DualHeadHybridCNNViT(
        num_classes=num_classes,
        efficientnet_pretrained=pretrained,
        img_size=img_size,
        vit_embed_dim=384,
        vit_num_heads=6,
        vit_num_layers=12,
        vit_mlp_ratio=4.0,
        bridge_dropout=0.1,
        vit_attn_dropout=0.1,
        vit_mlp_dropout=0.1,
        cnn_aux_dropout=0.3,
        mc_dropout_rate=mc_dropout_rate,
    )
