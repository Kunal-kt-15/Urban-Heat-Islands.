"""
Unit tests for U-Net architecture, tensor dimensions, and inference modes.
"""

import pytest
import torch
from models.unet import UNet


def test_unet_rgb_dimensions():
    """Verify input [B, 3, 256, 256] -> output [B, 4, 256, 256]."""
    batch_size = 2
    in_channels = 3
    num_classes = 4
    model = UNet(in_channels=in_channels, num_classes=num_classes, init_features=16)

    dummy_input = torch.randn(batch_size, in_channels, 256, 256)
    output = model(dummy_input)

    assert output.shape == (batch_size, num_classes, 256, 256), f"Expected shape {(batch_size, num_classes, 256, 256)}, got {output.shape}"


def test_unet_multispectral_rgb_nir():
    """Verify input [B, 4, 256, 256] (RGB+NIR) -> output [B, 4, 256, 256]."""
    batch_size = 2
    in_channels = 4
    num_classes = 4
    model = UNet(in_channels=in_channels, num_classes=num_classes, init_features=16)

    dummy_input = torch.randn(batch_size, in_channels, 256, 256)
    output = model(dummy_input)

    assert output.shape == (batch_size, num_classes, 256, 256)


def test_unet_softmax_and_mask_prediction():
    """Verify softmax probabilities sum to 1.0 per pixel and predict_mask returns [B, H, W]."""
    model = UNet(in_channels=3, num_classes=4, init_features=16)
    dummy_input = torch.randn(2, 3, 256, 256)

    probs = model.predict_proba(dummy_input)
    assert probs.shape == (2, 4, 256, 256)

    # Probabilities must sum to 1 across channel dimension
    prob_sums = probs.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)

    masks = model.predict_mask(dummy_input)
    assert masks.shape == (2, 256, 256)
    assert masks.dtype == torch.int64
    assert masks.max().item() < 4 and masks.min().item() >= 0
