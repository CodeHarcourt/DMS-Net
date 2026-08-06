"""
Main DMS-Net model definition.

This model performs multi-label classification from paired fundus images
with a shared backbone and cross-eye fusion modules.
"""

import torch
import torch.nn as nn
from torchvision import models

from .modules import OSIM, FeatureFusion
from .dmff import CASFM


class DualEfficientNet(nn.Module):
    """
    Dual-eye fundus multi-label classification network.

    Args:
        num_classes: Number of output classes. Defaults to 8.

    Inputs:
        left_img: Left-eye image tensor with shape [batch, 3, H, W].
        right_img: Right-eye image tensor with shape [batch, 3, H, W].

    Outputs:
        logits: Classification logits with shape [batch, num_classes].
    """
    def __init__(self, num_classes=8):
        super().__init__()
        # Extract shared backbone features for both eyes.
        resnet = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Preserve the spatial feature map.

        # Apply OmniPool integration before pairwise fusion.
        self.left_osim = OSIM(2048)
        self.right_osim = OSIM(2048)
        self.feature_fusion = FeatureFusion(dim_in=1024, dim_out=1024, reduction=True)

        # Aggregate the fused representation globally.
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Decode the pooled representation into class-specific queries.
        self.query_embeds = nn.Parameter(torch.Tensor(num_classes, 256))
        self.decoder = nn.MultiheadAttention(embed_dim=256, num_heads=8, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(1024*2, 256),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Map decoded queries to the final label space.
        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, num_classes)
        )

        # Fuse bilateral features with the architecture-defined CASFM module.
        self.casfm_fusion = CASFM(in_channels=1024, embed_dim=256)

        # Keep legacy attribute names available for downstream scripts.
        self.left_processor = self.left_osim
        self.right_processor = self.right_osim
        self.fusion_controller = self.feature_fusion
        self.dmff_fusion = self.casfm_fusion

        # Initialize learnable parameters.
        nn.init.xavier_normal_(self.query_embeds)
        nn.init.xavier_normal_(self.fc[0].weight)
        nn.init.xavier_normal_(self.classifier[1].weight)

    def forward(self, left_img, right_img):
        # Extract backbone features with shape [batch, 2048, 7, 7].
        left_feat = self.feature_extractor(left_img)
        right_feat = self.feature_extractor(right_img)

        # Integrate multi-scale spatial context for each eye.
        left_processed = self.left_osim(left_feat)
        right_processed = self.right_osim(right_feat)

        # Perform calibrated cross-eye semantic fusion.
        casfm_fused = self.casfm_fusion(left_processed, right_processed)

        # Preserve the original pairwise fusion pathway.
        original_fused = self.feature_fusion(left_processed, right_processed)

        # Concatenate both fusion paths before classification.
        combined_fused = torch.cat([original_fused, casfm_fused], dim=1)

        # Pool the final feature tensor to [batch, 2048].
        global_feat = self.avgpool(combined_fused).flatten(1)

        # Decode class queries from the pooled representation.
        projected = self.fc(global_feat)
        queries = self.query_embeds.unsqueeze(0).expand(projected.size(0), -1, -1)
        attn_output, _ = self.decoder(
            query=queries,
            key=projected.unsqueeze(1),
            value=projected.unsqueeze(1)
        )
        return self.classifier(attn_output).mean(dim=1)
