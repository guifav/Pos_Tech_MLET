import pytest
import torch
import pandas as pd
from unittest.mock import patch, MagicMock

from app.model.lstm_params import LSTMParams
from app.model.lstm import LSTMFactory


@pytest.fixture
def lstm_params():
    return LSTMParams(input_size=5, hidden_size=8, num_layers=1, output_size=2)


@pytest.fixture
def layer_config():
    return {"lstm1": "LSTM", "linear1": "Linear", "softmax1": "Softmax"}


@pytest.fixture
def lstm_factory(lstm_params, layer_config):
    return LSTMFactory(layer_config, lstm_params)


@pytest.fixture
def lstm_model(lstm_factory):
    return lstm_factory.create()


@pytest.fixture
def synthetic_input():
    # batch_size=4, seq_len=7, input_size=5
    return torch.randn(4, 7, 5)


@pytest.fixture
def mock_yfinance(monkeypatch):
    def fake_history(self, period):
        # 10 days of fake data
        idx = pd.date_range("2024-01-01", periods=10)
        df = pd.DataFrame({
            'High': [10,11,12,13,14,15,16,17,18,19],
            'Low': [5,6,7,8,9,10,11,12,13,14],
            'Close': [7,8,9,10,11,12,13,14,15,16],
            'Volume': [100,110,120,130,140,150,160,170,180,190]
        }, index=idx)
        return df
    monkeypatch.setattr('yfinance.Ticker.history', fake_history)


@pytest.fixture
def mock_mlflow(monkeypatch):
    monkeypatch.setattr('mlflow.log_param', lambda *a, **kw: None)
    monkeypatch.setattr('mlflow.log_metric', lambda *a, **kw: None)
    class DummyRun:
        def __enter__(self): return self
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    monkeypatch.setattr('mlflow.start_run', lambda *a, **kw: DummyRun())
