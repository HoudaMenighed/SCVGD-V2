import torch
import torch.nn as nn
import torch.nn.functional as F

class VisibilityHead(nn.Module):
    """
    Predicts visibility score for each patch token
    """

    def __init__(self, embed_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, patch_tokens):
        """
        patch_tokens: [B, N, D]
        return: visibility scores [B, N, 1]
        """
        v = self.mlp(patch_tokens)
        v = torch.sigmoid(v)
        return v


class ViTBranch(nn.Module):
    """
    Transformer branch with visibility modeling
    """

    def __init__(self, vit_model, embed_dim=384):
        super().__init__()
        self.vit = vit_model  # pre-trained ViT / DeiT
        self.embed_dim = embed_dim

        self.visibility_head = VisibilityHead(embed_dim)

        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        """
        x: input image [B, 3, H, W]
        """
        # ViT forward (example for timm ViT)
        features = self.vit.forward_features(x)

        # CLS token
        cls_token = features[:, 0]  # [B, D]

        # Patch tokens
        patch_tokens = features[:, 1:]  # [B, N, D]

        # Predict patch visibility
        visibility = self.visibility_head(patch_tokens)  # [B, N, 1]

        # Visibility-aware global feature
        weighted_patches = patch_tokens * visibility
        vis_global = weighted_patches.sum(dim=1) / (visibility.sum(dim=1) + 1e-6)

        vis_global = self.proj(vis_global)

        return {
            "cls_token": cls_token,
            "patch_tokens": patch_tokens,
            "visibility": visibility,
            "vis_global": vis_global
        }

x = torch.randn(2, 3, 256, 128)
out = ViTBranch(x)

print(out["vis_global"].shape)   # [2, D]
print(out["visibility"].shape)   # [2, N, 1]
