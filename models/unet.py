"""
U-Net Semantic Segmentation Architecture for Satellite Land-Cover Classification.
Supports arbitrary input channels (e.g., 3 for RGB, 4 for RGB+NIR) and 4 output classes.
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Standard Double Convolution Block:
    [Conv2d -> BatchNorm2d -> ReLU] x 2 with optional dropout.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout_p: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(p=dropout_p))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    Encoder-Decoder U-Net with skip connections for multi-class land cover segmentation.

    Args:
        in_channels: Number of input raster channels (3 for RGB, 4 for RGB+NIR).
        num_classes: Number of target segmentation classes (default 4).
        init_features: Number of base feature channels in first layer (default 32).
        dropout_p: Dropout probability in bottleneck (default 0.1).
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 4,
        init_features: int = 32,
        dropout_p: float = 0.1
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Encoder (Contracting Path)
        self.enc1 = DoubleConv(in_channels, init_features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = DoubleConv(init_features, init_features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = DoubleConv(init_features * 2, init_features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = DoubleConv(init_features * 4, init_features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = DoubleConv(init_features * 8, init_features * 16, dropout_p=dropout_p)

        # Decoder (Expanding Path)
        self.upconv4 = nn.ConvTranspose2d(init_features * 16, init_features * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(init_features * 16, init_features * 8)

        self.upconv3 = nn.ConvTranspose2d(init_features * 8, init_features * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(init_features * 8, init_features * 4)

        self.upconv2 = nn.ConvTranspose2d(init_features * 4, init_features * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(init_features * 4, init_features * 2)

        self.upconv1 = nn.ConvTranspose2d(init_features * 2, init_features, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(init_features * 2, init_features)

        # Final Classifier 1x1 Conv
        self.final_conv = nn.Conv2d(init_features, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Raw logits of shape (B, num_classes, H, W)
        """
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder with Skip Connections
        d4 = self.upconv4(b)
        # Handle spatial dimension matching if necessary
        if d4.shape != e4.shape:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode="bilinear", align_corners=True)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.upconv3(d4)
        if d3.shape != e3.shape:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode="bilinear", align_corners=True)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.upconv2(d3)
        if d2.shape != e2.shape:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=True)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.upconv1(d2)
        if d1.shape != e1.shape:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=True)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        logits = self.final_conv(d1)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference mode: returns softmax class probabilities (B, num_classes, H, W).
        """
        self.eval()
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference mode: returns predicted class indices (B, H, W).
        """
        self.eval()
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)
