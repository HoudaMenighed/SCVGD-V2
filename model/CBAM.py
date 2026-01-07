import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

class ChannelAttentionModule(nn.Module):

    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttentionModule, self).__init__()
        # Global pooling operations
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling

        # Shared MLP for dimensionality reduction and feature extraction
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        # Global average pooling and max pooling
        avg_out = self.avg_pool(x).view(batch_size, channels)
        max_out = self.max_pool(x).view(batch_size, channels)

        # Process through the shared FC layers
        avg_out = self.fc(avg_out).view(batch_size, channels, 1, 1)
        max_out = self.fc(max_out).view(batch_size, channels, 1, 1)

        # Generate channel attention map MC = σ(MLP(AvgPool(X)) + MLP(MaxPool(X)))
        MC = self.sigmoid(avg_out + max_out)

        # Return MC directly without applying it to the input
        # This follows the architecture where MC is applied separately
        return MC


class SpatialAttentionModule(nn.Module):
    """
    https://www.mdpi.com/2072-4292/12/1/188
    """
    def __init__(self, in_channels, feature_dim):
        super(SpatialAttentionModule, self).__init__()

        # Channel attention module to generate MC
        self.channel_attention = ChannelAttentionModule(in_channels)

        # 3×3 convolution for spatial attention map MS (F3×3×3)
        self.conv = nn.Conv2d(2, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

        # CNN feature extractor for refined features
        # Refines the attention-weighted features using two convolutional layers.)
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # Global pooling for vector representation
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Final embedding layers for ReID feature extraction
        self.fc_layers = nn.Sequential(
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, feature_dim),
            nn.BatchNorm1d(feature_dim)
        )

    def forward(self, x):
        # Apply channel attention to get MC
        MC = self.channel_attention(x)

        # Compute XC = X ⊗ MC (equation 8)
        XC = x * MC

        # Generate spatial attention features - Max_pool and Avg_pool operations
        # Average pooling along channel dimension
        avg_out = torch.mean(XC, dim=1, keepdim=True)  # [B, 1, H, W]

        # Max pooling along channel dimension
        max_out, _ = torch.max(XC, dim=1, keepdim=True)  # [B, 1, H, W]

        # Concatenate pooled features along channel dimension [B, 2, H, W]
        concat = torch.cat([avg_out, max_out], dim=1)

        # Apply convolution F3×3×3 and sigmoid to get spatial attention map MS (equation 9)
        MS = self.sigmoid(self.conv(concat))  # [B, 1, H, W]

        # Compute Y = XC ⊗ MS (equation 10) - element-wise multiplication
        Y = XC * MS  # Broadcast MS across all channels

        # Apply CNN layers for feature extraction
        Y = self.conv_layers(Y)  # Refines the attention-weighted features using two convolutional layers.

        # Global pooling converts spatial features to a vector
        Y = self.global_avg_pool(Y)
        Y = Y.view(Y.size(0), -1)

        # FC layers produce the final embedding
        Y = self.fc_layers(Y)

        # L2 normalization makes features suitable for similarity comparison
        Y = F.normalize(Y, p=2, dim=1)

        return Y