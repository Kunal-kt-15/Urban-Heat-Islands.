"""
Unit tests for LST Regression ANN architecture, batching, and feature dimensions.
"""

import pytest
import numpy as np
import torch
from models.regression_ann import LSTRegressionANN


def test_regression_dimensions():
    """Verify input [B, 7] -> output [B, 1]."""
    batch_size = 8
    in_features = 7
    model = LSTRegressionANN(in_features=in_features, hidden_layers=[32, 16])

    dummy_input = torch.randn(batch_size, in_features)
    output = model(dummy_input)

    assert output.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {output.shape}"


def test_regression_single_vector_handling():
    """Verify that passing 1D tensor [7] automatically handles batching and returns [1, 1]."""
    model = LSTRegressionANN(in_features=7, hidden_layers=[32, 16])
    model.eval()  # BatchNorm requires eval mode for single-sample inference
    dummy_input = torch.randn(7)

    output = model(dummy_input)
    assert output.shape == (1, 1)


def test_regression_predict_numpy():
    """Verify numpy inference method returns 1D array of predictions."""
    model = LSTRegressionANN(in_features=7, hidden_layers=[32, 16])
    dummy_np = np.random.randn(5, 7).astype(np.float32)

    preds = model.predict(dummy_np)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (5,)
