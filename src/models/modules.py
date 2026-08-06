"""
Core model modules used by DMS-Net.

This file contains the OmniPool spatial integration block and the
feature fusion utilities shared by the main network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OSIM(nn.Module):
    """OmniPool Spatial Integrator Module."""
    def __init__(self, in_channels):
        super().__init__()
        self.channel_compression = nn.Sequential(
            nn.Conv2d(5 * in_channels, in_channels // 2, 1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU()
        )
        # Keep the spatial gate aligned with the compressed feature width.
        self.spatial_att = nn.Sequential(
            nn.Conv2d(in_channels // 2, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Preserve the original spatial resolution for all pooled branches.
        gap = F.adaptive_avg_pool2d(x, (1, 1)).expand_as(x)
        gmp = F.adaptive_max_pool2d(x, (1, 1)).expand_as(x)
        spp_2x2 = F.adaptive_avg_pool2d(x, (2, 2))
        spp_2x2 = F.interpolate(spp_2x2, size=x.shape[2:], mode='bilinear', align_corners=False)
        spp_4x4 = F.adaptive_avg_pool2d(x, (4, 4))
        spp_4x4 = F.interpolate(spp_4x4, size=x.shape[2:], mode='bilinear', align_corners=False)

        combined = torch.cat([x, gap, gmp, spp_2x2, spp_4x4], dim=1)
        compressed = self.channel_compression(combined)

        spatial_weight = self.spatial_att(compressed)
        return compressed * spatial_weight


class DenseCatAdd(nn.Module):
    """Dense additive fusion block."""
    def __init__(self, in_chn, out_chn):
        super().__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)

        return self.conv_out(x1 + x2 + x3 + y1 + y2 + y3)


class DenseCatDiff(nn.Module):
    """Dense absolute-difference fusion block."""
    def __init__(self, in_chn, out_chn):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, in_chn, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
        )
        self.conv_out = torch.nn.Sequential(
            torch.nn.Conv2d(in_chn, out_chn, kernel_size=1, padding=0),
            torch.nn.BatchNorm2d(out_chn),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x, y):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2 + x1)

        y1 = self.conv1(y)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2 + y1)
        out = self.conv_out(torch.abs(x1 + x2 + x3 - y1 - y2 - y3))
        return out


class FeatureFusion(nn.Module):
    """Feature fusion head for paired ocular features."""
    def __init__(self, dim_in, dim_out, reduction=True):
        super().__init__()
        if reduction:
            self.reduction = torch.nn.Sequential(
                torch.nn.Conv2d(dim_in, dim_in // 2, kernel_size=1, padding=0),
                nn.BatchNorm2d(dim_in // 2),
                torch.nn.ReLU(inplace=True),
            )
            dim_in = dim_in // 2
        else:
            self.reduction = None
        self.cat1 = DenseCatAdd(dim_in, dim_out)
        self.cat2 = DenseCatDiff(dim_in, dim_out)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim_out, 2 * dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * dim_out),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim_out, 2 * dim_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(2 * dim_out),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Conv2d(2 * 1024, 512, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv2d(512, 1024, kernel_size=1)
        )

    def forward(self, x1, x2):
        if self.reduction is not None:
            x1 = self.reduction(x1)
            x2 = self.reduction(x2)
        x_add = self.cat1(x1, x2)
        x_diff = self.cat2(x1, x2)
        y = self.conv1(x_diff) + self.conv1(x_add)
        return self.fc(y)


# Backward-compatible aliases for existing imports.
MultiScaleFusion = OSIM
densecat_cat_add = DenseCatAdd
densecat_cat_diff = DenseCatDiff
