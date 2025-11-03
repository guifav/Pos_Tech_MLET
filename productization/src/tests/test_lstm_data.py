import pytest
import torch
from app.model.data import (
    NoProcessingSingle,
    NoProcessingMultiple,
    RangeSingle,
    RangeMultiple,
    RangeClusterMultiple
)


@pytest.mark.usefixtures("mock_yfinance", "mock_mlflow")
def test_no_processing_single():
    strat = NoProcessingSingle()
    X, y = strat.process(["FAKE"], "1y", seq_len=3)
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.shape[1] == 3  # seq_len
    assert y.shape[0] == X.shape[0]


@pytest.mark.usefixtures("mock_yfinance", "mock_mlflow")
def test_no_processing_multiple():
    strat = NoProcessingMultiple()
    X, y = strat.process(["A", "B"], "1y", seq_len=3)
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.shape[1] == 3
    assert y.shape[0] == X.shape[0]


@pytest.mark.usefixtures("mock_yfinance", "mock_mlflow")
def test_range_single():
    strat = RangeSingle()
    X, y = strat.process(["FAKE"], "1y", seq_len=3)
    assert X.shape[2] == 5  # 4 original + 1 range
    assert y.shape[0] == X.shape[0]


@pytest.mark.usefixtures("mock_yfinance", "mock_mlflow")
def test_range_multiple():
    strat = RangeMultiple()
    X, y = strat.process(["A", "B"], "1y", seq_len=3)
    # 4 features per ticker + 1 range per ticker = 10
    assert X.shape[2] == 10
    assert y.shape[0] == X.shape[0]


@pytest.mark.usefixtures("mock_yfinance", "mock_mlflow")
def test_range_cluster_multiple():
    strat = RangeClusterMultiple()
    X, y = strat.process(["A", "B"], "1y", seq_len=3)
    # 4 features per ticker + 1 range per ticker = 10, plus 1 cluster label
    assert X.shape[2] == 11
    assert y.shape[0] == X.shape[0]
