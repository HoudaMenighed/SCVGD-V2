import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

# attention module CBAM

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        b, c, h, w = x.size()

        # Channel attention
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        ca = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        ca = ca.view(b, c, 1, 1)
        x = x * ca

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * sa

        return x

# semantic head

class SemanticHead(nn.Module):
    def __init__(self, in_channels, num_parts):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_parts, kernel_size=1)

    def forward(self, x):
        """
        x: feature map [B, C, H, W]
        returns semantic maps [B, K, H, W]
        """
        return F.softmax(self.conv(x), dim=1)

# CNN branch : ResNet50
class CNNBranch(nn.Module):
    def __init__(self, num_parts=5, embed_dim=256):
        super().__init__()
        backbone = resnet50(pretrained=True)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.attention = CBAM(channels=1024)
        self.semantic_head = SemanticHead(2048, num_parts)

        self.global_proj = nn.Linear(2048, embed_dim)
        self.part_proj = nn.Linear(2048, embed_dim)

        self.num_parts = num_parts

    def forward(self, x):
        """
        x: input image [B, 3, H, W]
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.attention(x)

        x = self.layer4(x)  # [B, 2048, H', W']

        # Global feature
        global_feat = F.adaptive_avg_pool2d(x, 1).flatten(1)
        global_feat = self.global_proj(global_feat)

        # Semantic maps
        semantic_maps = self.semantic_head(x)  # [B, K, H', W']

        # Semantic part pooling
        part_feats = []
        for i in range(self.num_parts):
            mask = semantic_maps[:, i:i+1]
            feat = (x * mask).sum(dim=(2, 3)) / (mask.sum(dim=(2, 3)) + 1e-6)
            part_feats.append(self.part_proj(feat))

        part_feats = torch.stack(part_feats, dim=1)  # [B, K, D]

        return {
            "global_feat": global_feat,
            "part_feats": part_feats,
            "semantic_maps": semantic_maps
        }
