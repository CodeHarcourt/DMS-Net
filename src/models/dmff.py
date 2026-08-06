"""
Alignment and fusion modules used by the DMS-Net dual-branch backbone.

The naming in this file follows the architecture diagram:
- CASFM: Calibrated Analogous Semantic Fusion Module
- CIAM: Cross-Modal Integrative Alignment Module
- CCAM: Cross-Modal Contrastive Alignment Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedWeightPooling(nn.Module):
    """Shared pooling branch used before semantic fusion."""
    def __init__(self, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.lambda_param = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        x_avg = F.avg_pool2d(x, self.scale_factor)
        x_max = F.max_pool2d(x, self.scale_factor)
        return self.lambda_param * x_avg + (1 - self.lambda_param) * x_max


class CAFM(nn.Module):
    """Cross-attention feature modulation block."""
    def __init__(self, embed_dim):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value):
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        attn = torch.bmm(Q, K.transpose(1,2)) / (K.size(-1)**0.5)
        return self.out_proj(torch.bmm(F.softmax(attn, dim=-1), V))


class CCAM(nn.Module):
    """Cross-Modal Contrastive Alignment Module."""
    def __init__(self, embed_dim):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, 8)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim)
        )
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, src, aux):
        attn_out, _ = self.multihead_attn(aux, src, src)
        return self.alpha * attn_out + self.beta * src


class CIAM(nn.Module):
    """Cross-Modal Integrative Alignment Module."""
    def __init__(self, embed_dim):
        super().__init__()
        self.ciam_left = CCAM(embed_dim)
        self.ciam_right = CCAM(embed_dim)

    def forward(self, left_tokens, right_tokens):
        enhanced_right = self.ciam_right(right_tokens, left_tokens)
        enhanced_left = self.ciam_left(left_tokens, right_tokens)
        return enhanced_left, enhanced_right


class CASFM(nn.Module):
    """Calibrated Analogous Semantic Fusion Module."""
    def __init__(self, in_channels=1024, embed_dim=256):
        super().__init__()
        self.shared_pool = SharedWeightPooling(scale_factor=2)
        self.channel_adapter = nn.Conv2d(in_channels, embed_dim, 1)
        self.pos_embed = nn.Parameter(torch.randn(1, embed_dim, 1, 1))
        self.cross_modal_integrator = CIAM(embed_dim)
        self.nin_fusion = nn.Conv2d(embed_dim*2, in_channels, 1)
        # Use transposed convolution to restore the original feature scale.
        self.upsample = nn.ConvTranspose2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1
        )

    def forward(self, left_feat, right_feat):
        # Project the backbone features into the fusion embedding space.
        left_feat = self.channel_adapter(left_feat)
        right_feat = self.channel_adapter(right_feat)

        # Shrink the spatial resolution with shared pooling weights.
        left_pooled = self.shared_pool(left_feat)
        right_pooled = self.shared_pool(right_feat)

        # Serialize spatial features into token sequences.
        B, C, H, W = left_pooled.shape
        pos_embed = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear')
        pos_embed = pos_embed.view(1, C, -1).permute(0, 2, 1)

        left_tokens = left_pooled.flatten(2).permute(0, 2, 1) + pos_embed
        right_tokens = right_pooled.flatten(2).permute(0, 2, 1) + pos_embed

        # Align the paired branches before reconstructing the feature maps.
        left_enhanced, right_enhanced = self.cross_modal_integrator(left_tokens, right_tokens)

        # Rebuild the spatial feature tensors after token fusion.
        left_out = left_enhanced.permute(0, 2, 1).view(B, C, H, W)
        right_out = right_enhanced.permute(0, 2, 1).view(B, C, H, W)

        fused = self.nin_fusion(torch.cat([left_out, right_out], dim=1))
        return self.upsample(fused)


# Backward-compatible aliases for existing imports.
SpatialFeatureShrinking = SharedWeightPooling
CrossModalAttention = CAFM
CFEBlock = CCAM
ICFE = CIAM
DMFF = CASFM
