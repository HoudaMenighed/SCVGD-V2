import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchToPartVisibility(nn.Module):
    def __init__(self, num_parts, num_patches):
        super().__init__()
        self.num_parts = num_parts
        self.num_patches = num_patches

    def forward(self, patch_visibility):
        """
        patch_visibility: [B, N, 1]
        return: part_visibility [B, K]
        """
        B, N, _ = patch_visibility.shape
        patches_per_part = N // self.num_parts

        part_vis = []
        for i in range(self.num_parts):
            start = i * patches_per_part
            end = (i + 1) * patches_per_part if i < self.num_parts - 1 else N
            part_vis.append(patch_visibility[:, start:end].mean(dim=1))

        part_vis = torch.cat(part_vis, dim=1)
        return part_vis



class VisibilityFiLM(nn.Module):
    def __init__(self, num_parts, num_channels):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_parts, num_channels),
            nn.ReLU(inplace=True),
            nn.Linear(num_channels, num_channels)
        )

    def forward(self, feat_map, part_visibility):
        """
        feat_map: [B, C, H, W]
        part_visibility: [B, K]
        """
        gamma = torch.sigmoid(self.mlp(part_visibility))
        gamma = gamma.view(gamma.size(0), gamma.size(1), 1, 1)
        return feat_map * gamma


class VisibilityGuidedFusion(nn.Module):
    def __init__(self, embed_dim=256, num_parts=5, num_patches=128, cnn_channels=2048):
        super().__init__()

        self.patch_to_part = PatchToPartVisibility(num_parts, num_patches)
        self.film = VisibilityFiLM(num_parts, cnn_channels)

        fusion_dim = embed_dim * (2 + num_parts)
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.BatchNorm1d(fusion_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim // 2, embed_dim)
        )

    def forward(self, vit_out, cnn_out, cnn_feat_map=None):
        """
        vit_out: dict from ViTBranch
        cnn_out: dict from CNNBranch
        cnn_feat_map: optional feature map for FiLM
        """

        # Part-level visibility
        part_visibility = self.patch_to_part(vit_out["visibility"])

        # CNN semantic part gating
        part_feats = cnn_out["part_feats"] * part_visibility.unsqueeze(-1)

        # Optional FiLM modulation
        if cnn_feat_map is not None:
            cnn_feat_map = self.film(cnn_feat_map, part_visibility)

        # Concatenate features
        feats = [vit_out["vis_global"], cnn_out["global_feat"]]
        feats.append(part_feats.flatten(1))
        fused = torch.cat(feats, dim=1)

        embedding = self.fusion_fc(fused)
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding, part_visibility


