"""
Unit tests for LSTMFactory and LSTM model using synthetic data.
"""
import torch
import pytest
from app.model.lstm import LSTMFactory, LSTM


def test_lstm_factory_instantiation(lstm_params, layer_config):
    factory = LSTMFactory(layer_config, lstm_params)
    model = factory.create()
    assert isinstance(model, LSTM)
    assert hasattr(model, 'forward')
    assert len(model.layers) == len(layer_config)


def test_lstm_forward_pass(lstm_model, synthetic_input, lstm_params):
    output = lstm_model(synthetic_input)
    assert output.shape[0] == synthetic_input.shape[0]  # batch size
    assert output.shape[1] == lstm_params.output_size  # output size
    # Softmax output sums to 1
    assert torch.allclose(output.sum(dim=1), torch.ones(output.shape[0]), atol=1e-5)


def test_lstm_factory_invalid_layer(lstm_params):
    layer_config = {"lstm1": "LSTM", "invalid": "NotALayer"}
    factory = LSTMFactory(layer_config, lstm_params)
    with pytest.raises(ValueError):
        factory.create()


def test_lstm_factory_multiple_lstm_layers(lstm_params):
    # Test with two LSTM layers in sequence
    layer_config = {"lstm1": "LSTM", "lstm2": "LSTM", "linear1": "Linear", "softmax1": "Softmax"}
    factory = LSTMFactory(layer_config, lstm_params)
    model = factory.create()
    x = torch.randn(2, 7, lstm_params.input_size)
    output = model(x)
    assert output.shape[0] == 2
    assert output.shape[1] == lstm_params.output_size


def test_lstm_factory_sigmoid_layer(lstm_params):
    # Test with Sigmoid activation
    layer_config = {"lstm1": "LSTM", "linear1": "Linear", "sigmoid1": "Sigmoid"}
    factory = LSTMFactory(layer_config, lstm_params)
    model = factory.create()
    x = torch.randn(3, 5, lstm_params.input_size)
    output = model(x)
    assert output.shape[0] == 3
    assert output.shape[1] == lstm_params.output_size
    assert (output >= 0).all() and (output <= 1).all()
