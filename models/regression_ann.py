"""
Deep Feedforward Neural Network (ANN) for Land Surface Temperature (LST) Anomaly Regression.
Predicts LST anomaly based on land-cover percentages and environmental variables.
"""

from typing import List, Union
import numpy as np
import torch
import torch.nn as nn


class LSTRegressionANN(nn.Module):
    """
    Multilayer Perceptron for LST Anomaly Prediction.

    Input Features (7 dims):
        1. Impervious surface percentage
        2. Vegetation percentage
        3. Bare soil percentage
        4. Water percentage
        5. Elevation (m)
        6. Baseline Relative Humidity (%)
        7. Solar Irradiance (W/m^2)

    Target:
        Predicted Land Surface Temperature (LST) Anomaly in °C.
    """

    def __init__(
        self,
        in_features: int = 7,
        hidden_layers: List[int] = None,
        dropout_p: float = 0.15
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        self.in_features = in_features
        layers = []
        prev_dim = in_features

        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p > 0.0:
                layers.append(nn.Dropout(p=dropout_p))
            prev_dim = h_dim

        # Final regression output layer (1 continuous value)
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, in_features)
        Returns:
            Tensor of shape (B, 1)
        """
        # If single sample is passed without batch dimension (in_features,), unsqueeze to (1, in_features)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.network(x)

    @torch.no_grad()
    def predict(self, x: Union[np.ndarray, torch.Tensor], device: torch.device = None) -> np.ndarray:
        """
        Inference mode for single vector or batch of vectors.
        Returns:
            np.ndarray of shape (B,) with predicted LST anomalies.
        """
        self.eval()
        if isinstance(x, np.ndarray):
            tensor_x = torch.from_numpy(x).float()
        else:
            tensor_x = x.float()

        if tensor_x.dim() == 1:
            tensor_x = tensor_x.unsqueeze(0)

        if device is not None:
            tensor_x = tensor_x.to(device)
            self.to(device)

        preds = self.forward(tensor_x)
        return preds.squeeze(-1).cpu().numpy()
