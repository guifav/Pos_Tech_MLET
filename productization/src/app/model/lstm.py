"""
lstm.py

This module provides a flexible and extensible LSTM model construction utility for time series and sequence modeling, using the Factory Method design pattern.
The LSTMFactory class allows dynamic configuration of LSTM architectures for different use cases, such as stock price prediction,
by specifying the sequence of layers and their types.

The LSTMParams Pydantic model (imported) is used to validate and encapsulate the hyperparameters required for model instantiation.
"""

import torch.nn as nn
from app.model.lstm_params import LSTMParams


class LSTM(nn.Module):
    """
    Flexible LSTM model that applies a sequence of layers as defined by the factory.

    Args:
        layers (list[nn.Module]): List of PyTorch layers to be applied sequentially in the forward pass.

    Usage:
        Do not instantiate directly. Use LSTMFactory to create models with the desired architecture.
    """
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.LSTM):
                x, _ = layer(x)
                if i + 1 < len(self.layers) and not isinstance(self.layers[i + 1], nn.LSTM):
                    x = x[:, -1, :]
            else:
                x = layer(x)
        return x


class LSTMFactory:
    """
    Factory for creating LSTM models with customizable internal architectures.

    Args:
        layer_config (dict): Dictionary mapping layer keys to layer type names (e.g., {"lstm1": "LSTM", "linear1": "Linear"}).
        params (LSTMParams): Pydantic model containing LSTM hyperparameters.

    Methods:
        create(): Returns an LSTM model instance with the specified architecture.
    """
    def __init__(self, layer_config: dict, params: LSTMParams):
        """
        Initializes the LSTM model with a custom layer configuration and parameter
        set.
        Args:
            layer_config (dict): Mapping that describes each layer in the network
                (e.g., number of units, activation functions, and other layer‐specific
                hyperparameters).
            params (LSTMParams): Consolidated object containing global hyperparameters
                for model training and inference such as learning rate, dropout rate,
                number of epochs, etc.
        """
        self.layer_config = layer_config
        self.params = params

    def get_layer(self, layer_name: str, input_size=None):
        """
        Instantiate and return a specific PyTorch layer based on the provided
        ``layer_name``. Optionally override input_size for LSTM/Linear layers.
        """
        if input_size is None:
            input_size = self.params.input_size
        match layer_name:
            case 'LSTM':
                return nn.LSTM(
                    input_size,
                    self.params.hidden_size,
                    self.params.num_layers,
                    batch_first=self.params.batch_first
                )
            case 'Sigmoid':
                return nn.Sigmoid()
            case 'Linear':
                return nn.Linear(input_size, self.params.output_size)
            case 'Softmax':
                return nn.Softmax(dim=1)
            case _:
                raise ValueError(f"Layer {layer_name} not supported.")

    def create(self):
        """Builds and returns an LSTM model based on the current layer configuration."""
        layers = []
        current_input_size = self.params.input_size
        for layer_type in self.layer_config.values():
            if layer_type == 'LSTM':
                layer = self.get_layer('LSTM', input_size=current_input_size)
                current_input_size = self.params.hidden_size
            elif layer_type == 'Linear':
                layer = self.get_layer('Linear', input_size=current_input_size)
                current_input_size = self.params.output_size
            else:
                layer = self.get_layer(layer_type)
            layers.append(layer)
        return LSTM(layers)
