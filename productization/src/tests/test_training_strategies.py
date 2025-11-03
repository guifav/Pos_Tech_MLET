import pytest

from app.train.model import (
    NoProcessingSimpleStrategy,
    NoProcessingSingleStrategy,
    NoProcessingMultipleStrategy,
    RangeSingleStrategy,
    RangeMultipleStrategy,
    RangeClusterComplexStrategy,
    RangeClusterMultipleStrategy
)
from app.model.data import (
    NoProcessingSingle,
    NoProcessingMultiple,
    RangeSingle,
    RangeMultiple,
    RangeClusterMultiple,
)
from app.model.lstm import LSTMFactory

@pytest.fixture
def base_args():
    return {
        "tickers": ["A"],
        "period": "1y",
        "seq_len": 5,
        "num_epochs": 1,
        "learning_rate": 0.01,
        "batch_size": 2,
    }

strategies = [
    (
        NoProcessingSimpleStrategy,
        "NoProcessingSimple",
        NoProcessingMultiple,
    ),
    (
        NoProcessingSingleStrategy,
        "NoProcessingSingle",
        NoProcessingSingle,
    ),
    (
        NoProcessingMultipleStrategy,
        "NoProcessingMultiple",
        NoProcessingMultiple,
    ),
    (
        RangeSingleStrategy,
        "RangeSingle",
        RangeSingle,
    ),
    (
        RangeMultipleStrategy,
        "RangeMultiple",
        RangeMultiple,
    ),
    (
        RangeClusterComplexStrategy,
        "RangeClusterComplex",
        RangeClusterMultiple,
    ),
    (
        RangeClusterMultipleStrategy,
        "RangeClusterMultiple",
        RangeClusterMultiple,
    ),
]

@pytest.mark.parametrize("StrategyClass, expected_name, expected_datastrategy", strategies)
def test_strategy_interface_and_pipeline(
    StrategyClass,
    expected_name,
    expected_datastrategy,
    base_args,
    layer_config,
    lstm_params
):
    # instantiate strategy
    strat = StrategyClass(
        **base_args,
        layer_config=layer_config,
        lstm_params=lstm_params,
    )
    # test name
    assert strat.name == expected_name
    # test params
    params = strat.get_training_params()
    for key in base_args:
        assert params[key] == base_args[key]
    # test data pipeline strategy
    pipeline = strat.get_data_pipeline()
    assert hasattr(pipeline, 'strategy')
    assert isinstance(pipeline.strategy, expected_datastrategy)
    # test model factory
    factory = strat.get_model_factory()
    assert isinstance(factory, LSTMFactory)
    # factory holds correct configuration
    assert factory.layer_config is layer_config
    assert factory.params is lstm_params
