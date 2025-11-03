"""
data.py

Implements a flexible data pipeline for time series modeling using the Strategy design pattern. 
Provides multiple strategies for ingesting and processing stock price data from yfinance, including data cleaning, feature engineering, clustering, and MLflow tracking. 
The output is a PyTorch DataLoader compatible with LSTM models from lstm.py.
"""
from abc import ABC, abstractmethod

import mlflow
import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset

from app.model.lstm import LSTM


class DataStrategy(ABC):
    """Abstract base class for data processing strategies.

    Defines the interface for processing stock data into PyTorch tensors for LSTM models.
    """
    @abstractmethod
    def process(self, tickers: list[str], period: str, seq_len: int):
        """Process stock data and return PyTorch tensors.

        Args:
            tickers (list[str]): List of stock ticker symbols.
            period (str): Period string for yfinance (e.g., '1y').
            seq_len (int): Sequence length for LSTM input.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Feature and target tensors.
        """

    @staticmethod
    def load_data(tickers: list[str], period: str) -> pd.DataFrame:
        """Load historical stock data for given tickers and period.

        Args:
            tickers (list[str]): List of stock ticker symbols.
            period (str): Period string for yfinance.

        Returns:
            pd.DataFrame: DataFrame with concatenated stock data for all tickers.
        """
        dfs = []
        for t in tickers:
            hist = yf.Ticker(t).history(period=period)
            hist = hist[['High', 'Low', 'Close', 'Volume']].rename(columns=lambda c: f"{t}_{c}")
            dfs.append(hist)
        df = pd.concat(dfs, axis=1).dropna()
        return df

    @staticmethod
    def create_sequences(data: np.ndarray, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert array data into sequences and targets for LSTM.

        Args:
            data (np.ndarray): Input data array.
            seq_len (int): Sequence length for LSTM input.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Feature and target tensors.
        """
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len, :])
            # predict next day closing price of first ticker
            y.append(data[i+seq_len, 2])
        X = torch.tensor(np.stack(X), dtype=torch.float32)
        y = torch.tensor(np.stack(y).reshape(-1, 1), dtype=torch.float32)
        return X, y


class NoProcessingSingle(DataStrategy):
    """Strategy 1: No processing, single ticker.

    Loads and returns raw data for a single ticker as LSTM-ready tensors.
    """
    def process(self, tickers, period, seq_len):
        """See base class."""
        df = self.load_data([tickers[0]], period)
        arr = df.values
        return self.create_sequences(arr, seq_len)


class NoProcessingMultiple(DataStrategy):
    """Strategy 2: No processing, multiple tickers.

    Loads and returns raw data for multiple tickers as LSTM-ready tensors.
    """
    def process(self, tickers, period, seq_len):
        """See base class."""
        df = self.load_data(tickers, period)
        arr = df.values
        return self.create_sequences(arr, seq_len)


class RangeSingle(DataStrategy):
    """Strategy 3: Add daily range (High-Low) for single ticker.

    Adds a feature for daily price range to the data for a single ticker.
    """
    def process(self, tickers, period, seq_len):
        """See base class."""
        df = self.load_data([tickers[0]], period)
        df[f'{tickers[0]}_Range'] = df[f'{tickers[0]}_High'] - df[f'{tickers[0]}_Low']
        df = df.dropna()
        arr = df.values
        return self.create_sequences(arr, seq_len)


class RangeMultiple(DataStrategy):
    """Strategy 4: Add daily range (High-Low) for multiple tickers.

    Adds a feature for daily price range to the data for each ticker.
    """
    def process(self, tickers, period, seq_len):
        """See base class."""
        df = self.load_data(tickers, period)
        for t in tickers:
            df[f'{t}_Range'] = df[f'{t}_High'] - df[f'{t}_Low']
        df = df.dropna()
        arr = df.values
        return self.create_sequences(arr, seq_len)


class RangeClusterMultiple(DataStrategy):
    """Strategy 5: Add daily range and DBSCAN clustering for multiple tickers.

    Adds a feature for daily price range and a cluster label (DBSCAN) to the data for each ticker. Logs clustering parameters and metrics to MLflow.
    """
    def process(self, tickers, period, seq_len):
        """See base class. Logs DBSCAN parameters and metrics to MLflow."""
        df = self.load_data(tickers, period)
        for t in tickers:
            df[f'{t}_Range'] = df[f'{t}_High'] - df[f'{t}_Low']
        df = df.dropna()
        # clustering: detect outlier days
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df.values)
        db = DBSCAN(eps=0.5, min_samples=5).fit(scaled)
        df['Cluster'] = db.labels_
        mlflow.log_param('dbscan_eps', db.eps)
        mlflow.log_param('dbscan_min_samples', db.min_samples)
        labels = db.labels_
        n_noise = int((labels == -1).sum())
        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        mlflow.log_metric('dbscan_num_clusters', n_clusters)
        mlflow.log_metric('dbscan_noise_points', n_noise)
        arr = df.values
        return self.create_sequences(arr, seq_len)


class DataPipeline:
    """Context class for running the data pipeline with a chosen strategy.

    Handles the execution of a data processing strategy and returns a DataLoader for LSTM training. Logs parameters and metrics to MLflow.
    """
    def __init__(self, strategy: DataStrategy, model: LSTM, batch_size: int = 32):
        """Initializes the DataPipeline.

        Args:
            strategy (DataStrategy): The data processing strategy to use.
            model (LSTM): The LSTM model instance (for compatibility reference).
            batch_size (int, optional): Batch size for DataLoader. Defaults to 32.
        """
        self.strategy = strategy
        self.model = model
        self.batch_size = batch_size

    def run(self, tickers: list[str], period: str, seq_len: int) -> DataLoader:
        """Executes the pipeline and returns a DataLoader.

        Args:
            tickers (list[str]): List of stock ticker symbols.
            period (str): Period string for yfinance.
            seq_len (int): Sequence length for LSTM input.

        Returns:
            DataLoader: PyTorch DataLoader with processed data.
        """
        with mlflow.start_run():
            mlflow.log_param('strategy', type(self.strategy).__name__)
            mlflow.log_param('tickers', tickers)
            mlflow.log_param('period', period)
            mlflow.log_param('seq_len', seq_len)
            X, y = self.strategy.process(tickers, period, seq_len)
            mlflow.log_metric('num_samples', X.shape[0])
            dataset = TensorDataset(X, y)
            loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            return loader
