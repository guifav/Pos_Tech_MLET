"""
Module: lstm_params.py

This module defines the LSTMParams Pydantic model, which encapsulates the hyperparameters required for the construction of LSTM models in a flexible and validated manner. It is intended to be used in conjunction with LSTM model factories or other model-building utilities that require explicit hyperparameter management.
"""

from pydantic import BaseModel, Field


class LSTMParams(BaseModel):
    """
    Pydantic model for LSTM hyperparameters.

    Attributes:
        input_size (int): Number of features in the input data.
        hidden_size (int): Number of hidden units in the LSTM layer(s).
        num_layers (int): Number of stacked LSTM layers.
        output_size (int): Number of output units (e.g., for regression or classification).
        batch_first (bool): If True, input and output tensors are provided as (batch, seq, feature).
    """
    input_size: int = Field(..., description="Number of features in the input data.")
    hidden_size: int = Field(..., description="Number of hidden units in the LSTM layer(s).")
    num_layers: int = Field(..., description="Number of stacked LSTM layers.")
    output_size: int = Field(..., description="Number of output units (e.g., for regression or classification).")
    batch_first: bool = Field(True, description="If True, input and output tensors are provided as (batch, seq, feature).")
