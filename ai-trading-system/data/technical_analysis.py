import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union


class TechnicalAnalysisProcessor:
    """
    Processes OHLCV market data to calculate technical indicators and generate signals.
    """
    
    def __init__(self):
        """Initialize the technical analysis processor."""
        self.indicators = {}
        self.signals = {}
    
    def process_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process OHLCV data and calculate all technical indicators.
        
        Args:
            df: DataFrame with OHLCV data (must contain open, high, low, close, volume columns)
            
        Returns:
            DataFrame with added technical indicators
        """
        # Make a copy to avoid modifying the original dataframe
        data = df.copy()
        
        # Calculate moving averages
        data = self.calculate_sma(data, periods=[20, 50, 200])
        data = self.calculate_ema(data, periods=[12, 26, 50])
        
        # Calculate momentum indicators
        data = self.calculate_rsi(data, period=14)
        data = self.calculate_macd(data)
        
        # Calculate volatility indicators
        data = self.calculate_bollinger_bands(data, period=20, std_dev=2)
        data = self.calculate_atr(data, period=14)
        
        # Generate signals
        data = self.generate_signals(data)
        
        return data
    
    def calculate_sma(self, df: pd.DataFrame, periods: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """
        Calculate Simple Moving Averages for given periods.
        
        Args:
            df: DataFrame with price data
            periods: List of periods to calculate SMAs for
            
        Returns:
            DataFrame with added SMA columns
        """
        data = df.copy()
        for period in periods:
            data[f'sma_{period}'] = data['close'].rolling(window=period).mean()
        
        return data
    
    def calculate_ema(self, df: pd.DataFrame, periods: List[int] = [12, 26, 50]) -> pd.DataFrame:
        """
        Calculate Exponential Moving Averages for given periods.
        
        Args:
            df: DataFrame with price data
            periods: List of periods to calculate EMAs for
            
        Returns:
            DataFrame with added EMA columns
        """
        data = df.copy()
        for period in periods:
            data[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False).mean()
        
        return data
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index.
        
        Args:
            df: DataFrame with price data
            period: RSI calculation period
            
        Returns:
            DataFrame with added RSI column
        """
        data = df.copy()
        
        # Calculate price changes
        delta = data['close'].diff()
        
        # Create separate gains and losses series
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss over the specified period
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gain / avg_loss
        
        # Calculate RSI
        data['rsi'] = 100 - (100 / (1 + rs))
        
        return data
    
    def calculate_macd(self, df: pd.DataFrame, fast_period: int = 12, 
                      slow_period: int = 26, signal_period: int = 9) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            df: DataFrame with price data
            fast_period: Fast EMA period
            slow_period: Slow EMA period
            signal_period: Signal line EMA period
            
        Returns:
            DataFrame with added MACD columns
        """
        data = df.copy()
        
        # Calculate fast and slow EMAs
        fast_ema = data['close'].ewm(span=fast_period, adjust=False).mean()
        slow_ema = data['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        data['macd'] = fast_ema - slow_ema
        
        # Calculate signal line
        data['macd_signal'] = data['macd'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate histogram
        data['macd_hist'] = data['macd'] - data['macd_signal']
        
        return data
    
    def calculate_bollinger_bands(self, df: pd.DataFrame, period: int = 20, 
                                 std_dev: float = 2.0) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.
        
        Args:
            df: DataFrame with price data
            period: Moving average period
            std_dev: Number of standard deviations for bands
            
        Returns:
            DataFrame with added Bollinger Bands columns
        """
        data = df.copy()
        
        # Calculate the middle band (SMA)
        data['bb_middle'] = data['close'].rolling(window=period).mean()
        
        # Calculate the standard deviation
        rolling_std = data['close'].rolling(window=period).std()
        
        # Calculate the upper and lower bands
        data['bb_upper'] = data['bb_middle'] + (rolling_std * std_dev)
        data['bb_lower'] = data['bb_middle'] - (rolling_std * std_dev)
        
        # Calculate %B (relative position within the bands)
        data['bb_pct_b'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        return data
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average True Range.
        
        Args:
            df: DataFrame with OHLC data
            period: ATR calculation period
            
        Returns:
            DataFrame with added ATR column
        """
        data = df.copy()
        
        # Calculate true range
        high_low = data['high'] - data['low']
        high_close_prev = abs(data['high'] - data['close'].shift(1))
        low_close_prev = abs(data['low'] - data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        
        # Calculate ATR
        data['atr'] = true_range.rolling(window=period).mean()
        
        return data
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on technical indicators.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            DataFrame with added signal columns
        """
        data = df.copy()
        
        # Moving Average Crossover signals
        data['signal_sma_20_50_cross'] = 0
        data.loc[data['sma_20'] > data['sma_50'], 'signal_sma_20_50_cross'] = 1
        data.loc[data['sma_20'] < data['sma_50'], 'signal_sma_20_50_cross'] = -1
        
        # RSI signals
        data['signal_rsi'] = 0
        data.loc[data['rsi'] < 30, 'signal_rsi'] = 1  # Oversold
        data.loc[data['rsi'] > 70, 'signal_rsi'] = -1  # Overbought
        
        # MACD signals
        data['signal_macd'] = 0
        data.loc[data['macd'] > data['macd_signal'], 'signal_macd'] = 1  # Bullish
        data.loc[data['macd'] < data['macd_signal'], 'signal_macd'] = -1  # Bearish
        
        # Bollinger Bands signals
        data['signal_bb'] = 0
        data.loc[data['close'] < data['bb_lower'], 'signal_bb'] = 1  # Price below lower band
        data.loc[data['close'] > data['bb_upper'], 'signal_bb'] = -1  # Price above upper band
        
        # Composite signal (simple average of all signals)
        signal_columns = [col for col in data.columns if col.startswith('signal_')]
        data['composite_signal'] = data[signal_columns].mean(axis=1)
        
        return data
    
    def prepare_rag_data(self, df: pd.DataFrame) -> Dict:
        """
        Prepare data for the RAG (Retrieval-Augmented Generation) system.
        
        Args:
            df: DataFrame with technical indicators and signals
            
        Returns:
            Dictionary with processed data for RAG
        """
        data = df.copy().tail(10)  # Get the most recent data
        
        # Create a summary of current market conditions
        latest = data.iloc[-1]
        
        # Convert timestamp to string format for JSON serialization
        timestamp = latest.name
        if hasattr(timestamp, 'isoformat'):
            timestamp = timestamp.isoformat()
        else:
            timestamp = str(timestamp)
        
        summary = {
            'timestamp': timestamp,
            'price': float(latest['close']),
            'indicators': {
                'sma_20': float(latest.get('sma_20')) if not pd.isna(latest.get('sma_20')) else None,
                'sma_50': float(latest.get('sma_50')) if not pd.isna(latest.get('sma_50')) else None,
                'sma_200': float(latest.get('sma_200')) if not pd.isna(latest.get('sma_200')) else None,
                'rsi': float(latest.get('rsi')) if not pd.isna(latest.get('rsi')) else None,
                'macd': float(latest.get('macd')) if not pd.isna(latest.get('macd')) else None,
                'macd_signal': float(latest.get('macd_signal')) if not pd.isna(latest.get('macd_signal')) else None,
                'bb_upper': float(latest.get('bb_upper')) if not pd.isna(latest.get('bb_upper')) else None,
                'bb_middle': float(latest.get('bb_middle')) if not pd.isna(latest.get('bb_middle')) else None,
                'bb_lower': float(latest.get('bb_lower')) if not pd.isna(latest.get('bb_lower')) else None,
                'atr': float(latest.get('atr')) if not pd.isna(latest.get('atr')) else None
            },
            'signals': {
                'ma_cross': int(latest.get('signal_sma_20_50_cross')) if not pd.isna(latest.get('signal_sma_20_50_cross')) else 0,
                'rsi': int(latest.get('signal_rsi')) if not pd.isna(latest.get('signal_rsi')) else 0,
                'macd': int(latest.get('signal_macd')) if not pd.isna(latest.get('signal_macd')) else 0,
                'bollinger': int(latest.get('signal_bb')) if not pd.isna(latest.get('signal_bb')) else 0,
                'composite': float(latest.get('composite_signal')) if not pd.isna(latest.get('composite_signal')) else 0
            },
            'trends': {
                'short_term': 'bullish' if latest.get('composite_signal', 0) > 0 else 'bearish',
                'price_vs_sma50': 'above' if latest['close'] > latest.get('sma_50', float('inf')) else 'below',
                'price_vs_sma200': 'above' if latest['close'] > latest.get('sma_200', float('inf')) else 'below',
                'volatility': 'high' if latest.get('atr') > data['atr'].mean() else 'normal'
            }
        }
        
        return summary
