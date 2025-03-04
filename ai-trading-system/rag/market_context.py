#!/usr/bin/env python3
"""
Market Context Module for AI Trading System.

This module provides functionality to gather and format multi-timeframe market data
for use in the RAG (Retrieval Augmented Generation) system. It creates and maintains
a database of market patterns and contexts that can be used for similarity searches.
"""

import json
import pandas as pd
import numpy as np
import os
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import local modules
from data.collector import MarketDataCollector
from data.technical_analysis import TechnicalAnalysisProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('market_context')

class MarketContext:
    """
    Gathers, processes, and stores market context information across multiple timeframes.
    
    This class is responsible for:
    1. Collecting market data for multiple timeframes
    2. Processing technical indicators
    3. Creating and storing market context snapshots
    4. Enabling similarity searches of historical contexts
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize the MarketContext module.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config = self._load_config(config_path)
        self.data_collector = MarketDataCollector(exchange_id=self.config.get("exchange", {}).get("id", "binance"), 
                                                 testnet=self.config.get("exchange", {}).get("testnet", True))
        self.ta_processor = TechnicalAnalysisProcessor()
        
        # Database paths
        self.context_db_path = Path(self.config.get("rag", {}).get("context_db_path", "data/context_db"))
        self.vector_db_path = Path(self.config.get("rag", {}).get("vector_db_path", "data/vector_db"))
        self.context_db_path.mkdir(parents=True, exist_ok=True)
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize databases
        self.context_db = self._load_context_db()
        self.vectorizer = self._initialize_vectorizer()
        
        # Timeframes to collect
        self.timeframes = self.config.get("rag", {}).get("timeframes", ["15m", "1h", "4h"])
        
        logger.info(f"MarketContext initialized with timeframes: {self.timeframes}")
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to the configuration file
            
        Returns:
            Dict containing configuration parameters
        """
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Error loading configuration: {e}. Using default values.")
            return {
                "exchange": {
                    "id": "binance",
                    "testnet": True
                },
                "rag": {
                    "context_db_path": "data/context_db",
                    "vector_db_path": "data/vector_db",
                    "timeframes": ["15m", "1h", "4h"],
                    "lookback_periods": 100,
                    "context_retention_days": 30
                }
            }
    
    def _load_context_db(self) -> Dict:
        """
        Load the context database from disk.
        
        Returns:
            Dict containing market context database
        """
        db_file = self.context_db_path / "context_db.pkl"
        if db_file.exists():
            try:
                with open(db_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error loading context database: {e}")
                return self._initialize_context_db()
        else:
            return self._initialize_context_db()
    
    def _initialize_context_db(self) -> Dict:
        """
        Initialize an empty context database.
        
        Returns:
            Dict containing initialized market context database
        """
        return {
            "version": 1.0,
            "created_at": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "contexts": [],
            "metadata": {
                "total_contexts": 0,
                "symbols": []
            }
        }
    
    def _initialize_vectorizer(self) -> TfidfVectorizer:
        """
        Initialize the TF-IDF vectorizer for context similarity search.
        
        Returns:
            TF-IDF vectorizer instance
        """
        vectorizer_file = self.vector_db_path / "vectorizer.pkl"
        vectors_file = self.vector_db_path / "context_vectors.pkl"
        
        if vectorizer_file.exists() and vectors_file.exists():
            try:
                with open(vectorizer_file, 'rb') as f:
                    vectorizer = pickle.load(f)
                with open(vectors_file, 'rb') as f:
                    self.context_vectors = pickle.load(f)
                return vectorizer
            except Exception as e:
                logger.error(f"Error loading vectorizer: {e}")
        
        # Initialize new vectorizer
        vectorizer = TfidfVectorizer(
            analyzer='word',
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 3)
        )
        self.context_vectors = None
        return vectorizer
    
    def _save_context_db(self) -> None:
        """Save the context database to disk."""
        db_file = self.context_db_path / "context_db.pkl"
        try:
            with open(db_file, 'wb') as f:
                pickle.dump(self.context_db, f)
            logger.info(f"Context database saved with {len(self.context_db['contexts'])} entries")
        except Exception as e:
            logger.error(f"Error saving context database: {e}")
    
    def _save_vectorizer(self) -> None:
        """Save the TF-IDF vectorizer and context vectors to disk."""
        vectorizer_file = self.vector_db_path / "vectorizer.pkl"
        vectors_file = self.vector_db_path / "context_vectors.pkl"
        
        try:
            with open(vectorizer_file, 'wb') as f:
                pickle.dump(self.vectorizer, f)
            
            if self.context_vectors is not None:
                with open(vectors_file, 'wb') as f:
                    pickle.dump(self.context_vectors, f)
            
            logger.info("Vectorizer and context vectors saved")
        except Exception as e:
            logger.error(f"Error saving vectorizer: {e}")
    
    def collect_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Collect market data for multiple timeframes.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            Dictionary mapping timeframes to DataFrames with market data
        """
        lookback = self.config.get("rag", {}).get("lookback_periods", 100)
        
        result = {}
        for timeframe in self.timeframes:
            try:
                df = self.data_collector.fetch_ohlcv(symbol, timeframe=timeframe, limit=lookback)
                if not df.empty:
                    # Process technical indicators
                    df = self.ta_processor.process_data(df)
                    result[timeframe] = df
                    logger.info(f"Collected {len(df)} {timeframe} candles for {symbol}")
                else:
                    logger.warning(f"No data returned for {symbol} on {timeframe} timeframe")
            except Exception as e:
                logger.error(f"Error collecting {timeframe} data for {symbol}: {e}")
        
        return result
    
    def create_market_snapshot(self, symbol: str, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Create a comprehensive market snapshot from multi-timeframe data.
        
        Args:
            symbol: Trading pair symbol
            market_data: Dictionary of DataFrames with market data for different timeframes
            
        Returns:
            Dictionary containing the market snapshot
        """
        if not market_data:
            logger.warning(f"No market data available for {symbol}")
            return {}
        
        # Initialize snapshot
        snapshot = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "timeframes": {},
            "aggregated_signals": {
                "short_term_trend": None,
                "medium_term_trend": None,
                "volatility": None,
                "momentum": None,
                "support_resistance": []
            },
            "patterns": []
        }
        
        # Process each timeframe
        for timeframe, df in market_data.items():
            if df.empty or len(df) < 10:
                continue
            
            # Get latest data point with indicators
            latest = df.iloc[-1]
            
            # Create timeframe-specific context
            tf_context = {
                "close": float(latest['close']),
                "volume": float(latest['volume']),
                "indicators": {
                    "sma_20": float(latest.get('sma_20', 0)) if not pd.isna(latest.get('sma_20', 0)) else None,
                    "sma_50": float(latest.get('sma_50', 0)) if not pd.isna(latest.get('sma_50', 0)) else None,
                    "sma_200": float(latest.get('sma_200', 0)) if not pd.isna(latest.get('sma_200', 0)) else None,
                    "ema_12": float(latest.get('ema_12', 0)) if not pd.isna(latest.get('ema_12', 0)) else None,
                    "ema_26": float(latest.get('ema_26', 0)) if not pd.isna(latest.get('ema_26', 0)) else None,
                    "rsi": float(latest.get('rsi', 0)) if not pd.isna(latest.get('rsi', 0)) else None,
                    "macd": float(latest.get('macd', 0)) if not pd.isna(latest.get('macd', 0)) else None,
                    "macd_signal": float(latest.get('macd_signal', 0)) if not pd.isna(latest.get('macd_signal', 0)) else None,
                    "macd_hist": float(latest.get('macd_hist', 0)) if not pd.isna(latest.get('macd_hist', 0)) else None,
                    "bb_upper": float(latest.get('bb_upper', 0)) if not pd.isna(latest.get('bb_upper', 0)) else None,
                    "bb_middle": float(latest.get('bb_middle', 0)) if not pd.isna(latest.get('bb_middle', 0)) else None,
                    "bb_lower": float(latest.get('bb_lower', 0)) if not pd.isna(latest.get('bb_lower', 0)) else None,
                    "atr": float(latest.get('atr', 0)) if not pd.isna(latest.get('atr', 0)) else None
                },
                "signals": {
                    "ma_cross": int(latest.get('signal_sma_20_50_cross', 0)) if not pd.isna(latest.get('signal_sma_20_50_cross', 0)) else 0,
                    "rsi": int(latest.get('signal_rsi', 0)) if not pd.isna(latest.get('signal_rsi', 0)) else 0,
                    "macd": int(latest.get('signal_macd', 0)) if not pd.isna(latest.get('signal_macd', 0)) else 0,
                    "bollinger": int(latest.get('signal_bb', 0)) if not pd.isna(latest.get('signal_bb', 0)) else 0,
                    "composite": float(latest.get('composite_signal', 0)) if not pd.isna(latest.get('composite_signal', 0)) else 0
                }
            }
            
            # Add previous data points for pattern detection
            recent_data = df.iloc[-5:].copy()
            candlesticks = []
            for idx, row in recent_data.iterrows():
                candle = {
                    "timestamp": idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume'])
                }
                candlesticks.append(candle)
            
            tf_context["recent_candles"] = candlesticks
            snapshot["timeframes"][timeframe] = tf_context
            
            # Identify technical patterns
            patterns = self._identify_patterns(df)
            if patterns:
                for pattern in patterns:
                    pattern["timeframe"] = timeframe
                    snapshot["patterns"].append(pattern)
        
        # Update aggregated signals
        if "1h" in snapshot["timeframes"]:
            snapshot["aggregated_signals"]["short_term_trend"] = self._determine_trend(
                snapshot["timeframes"]["1h"]["signals"]["composite"]
            )
        
        if "4h" in snapshot["timeframes"]:
            snapshot["aggregated_signals"]["medium_term_trend"] = self._determine_trend(
                snapshot["timeframes"]["4h"]["signals"]["composite"]
            )
        
        # Calculate volatility based on ATR
        if "1h" in snapshot["timeframes"] and snapshot["timeframes"]["1h"]["indicators"]["atr"] is not None:
            price = snapshot["timeframes"]["1h"]["close"]
            atr = snapshot["timeframes"]["1h"]["indicators"]["atr"]
            if price > 0:
                volatility_percentage = (atr / price) * 100
                if volatility_percentage < 1:
                    snapshot["aggregated_signals"]["volatility"] = "low"
                elif volatility_percentage < 3:
                    snapshot["aggregated_signals"]["volatility"] = "medium"
                else:
                    snapshot["aggregated_signals"]["volatility"] = "high"
        
        # Determine momentum
        if "1h" in snapshot["timeframes"]:
            rsi = snapshot["timeframes"]["1h"]["indicators"]["rsi"]
            macd = snapshot["timeframes"]["1h"]["indicators"]["macd"]
            if rsi is not None and macd is not None:
                if rsi > 70 and macd > 0:
                    snapshot["aggregated_signals"]["momentum"] = "strong_bullish"
                elif rsi > 60 and macd > 0:
                    snapshot["aggregated_signals"]["momentum"] = "bullish"
                elif rsi < 30 and macd < 0:
                    snapshot["aggregated_signals"]["momentum"] = "strong_bearish"
                elif rsi < 40 and macd < 0:
                    snapshot["aggregated_signals"]["momentum"] = "bearish"
                else:
                    snapshot["aggregated_signals"]["momentum"] = "neutral"
        
        # Find support and resistance levels
        if "4h" in market_data:
            support_resistance = self._identify_support_resistance(market_data["4h"])
            snapshot["aggregated_signals"]["support_resistance"] = support_resistance
        
        # Create a text description of the market context
        snapshot["text_description"] = self._create_context_description(snapshot)
        
        return snapshot
    
    def _determine_trend(self, composite_signal: float) -> str:
        """
        Determine trend based on composite signal.
        
        Args:
            composite_signal: Composite technical signal value
            
        Returns:
            Trend description string
        """
        if composite_signal > 0.7:
            return "strong_bullish"
        elif composite_signal > 0.3:
            return "bullish"
        elif composite_signal > -0.3:
            return "neutral"
        elif composite_signal > -0.7:
            return "bearish"
        else:
            return "strong_bearish"
    
    def _identify_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """
        Identify technical chart patterns in the data.
        
        Args:
            df: DataFrame with price and indicator data
            
        Returns:
            List of identified patterns with descriptions
        """
        if len(df) < 20:
            return []
        
        patterns = []
        
        # Get recent data
        recent = df.iloc[-20:]
        
        # Check for double top pattern
        highs = recent['high'].values
        if len(highs) > 10:
            # Find local maxima
            peaks = []
            for i in range(1, len(highs) - 1):
                if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                    peaks.append((i, highs[i]))
            
            # Check for two similar peaks
            for i in range(len(peaks) - 1):
                for j in range(i+1, len(peaks)):
                    price_diff_pct = abs(peaks[i][1] - peaks[j][1]) / peaks[i][1]
                    idx_diff = peaks[j][0] - peaks[i][0]
                    
                    if price_diff_pct < 0.02 and idx_diff > 3 and idx_diff < 15:
                        patterns.append({
                            "pattern": "double_top",
                            "confidence": 0.7,
                            "description": "Potential double top pattern detected, suggesting bearish reversal"
                        })
                        break
        
        # Check for bullish engulfing
        for i in range(1, len(recent) - 1):
            curr = recent.iloc[i]
            prev = recent.iloc[i-1]
            
            # Bearish candle followed by bullish engulfing
            if prev['close'] < prev['open'] and curr['close'] > curr['open']:
                if curr['open'] <= prev['close'] and curr['close'] > prev['open']:
                    patterns.append({
                        "pattern": "bullish_engulfing",
                        "confidence": 0.65,
                        "description": "Bullish engulfing pattern detected, suggesting potential reversal to upside"
                    })
        
        # Check for bearish engulfing
        for i in range(1, len(recent) - 1):
            curr = recent.iloc[i]
            prev = recent.iloc[i-1]
            
            # Bullish candle followed by bearish engulfing
            if prev['close'] > prev['open'] and curr['close'] < curr['open']:
                if curr['open'] >= prev['close'] and curr['close'] < prev['open']:
                    patterns.append({
                        "pattern": "bearish_engulfing",
                        "confidence": 0.65,
                        "description": "Bearish engulfing pattern detected, suggesting potential reversal to downside"
                    })
        
        # Check for golden cross (short-term MA crossing above long-term MA)
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            last_cross = None
            for i in range(1, min(20, len(recent))):
                curr = recent.iloc[-i]
                prev = recent.iloc[-(i+1)]
                
                if prev['sma_50'] <= prev['sma_200'] and curr['sma_50'] > curr['sma_200']:
                    patterns.append({
                        "pattern": "golden_cross",
                        "confidence": 0.8,
                        "description": "Golden cross detected (50 SMA crossing above 200 SMA), indicating potential bullish trend"
                    })
                    last_cross = "golden"
                    break
                
                if prev['sma_50'] >= prev['sma_200'] and curr['sma_50'] < curr['sma_200']:
                    patterns.append({
                        "pattern": "death_cross",
                        "confidence": 0.8,
                        "description": "Death cross detected (50 SMA crossing below 200 SMA), indicating potential bearish trend"
                    })
                    last_cross = "death"
                    break
        
        # Check for RSI divergence
        if 'rsi' in df.columns and len(recent) >= 10:
            prices = recent['close'].values
            rsi_values = recent['rsi'].values
            
            # Find local price extrema
            price_peaks = []
            price_troughs = []
            
            for i in range(2, len(prices) - 2):
                if prices[i] > prices[i-1] and prices[i] > prices[i-2] and prices[i] > prices[i+1] and prices[i] > prices[i+2]:
                    price_peaks.append((i, prices[i], rsi_values[i]))
                if prices[i] < prices[i-1] and prices[i] < prices[i-2] and prices[i] < prices[i+1] and prices[i] < prices[i+2]:
                    price_troughs.append((i, prices[i], rsi_values[i]))
            
            # Check for divergence
            if len(price_peaks) >= 2:
                if price_peaks[-1][1] > price_peaks[-2][1] and price_peaks[-1][2] < price_peaks[-2][2]:
                    patterns.append({
                        "pattern": "bearish_divergence",
                        "confidence": 0.75,
                        "description": "Bearish RSI divergence detected (higher highs in price, lower highs in RSI)"
                    })
            
            if len(price_troughs) >= 2:
                if price_troughs[-1][1] < price_troughs[-2][1] and price_troughs[-1][2] > price_troughs[-2][2]:
                    patterns.append({
                        "pattern": "bullish_divergence",
                        "confidence": 0.75,
                        "description": "Bullish RSI divergence detected (lower lows in price, higher lows in RSI)"
                    })
        
        return patterns
    
    def _identify_support_resistance(self, df: pd.DataFrame) -> List[Dict]:
        """
        Identify support and resistance levels from price data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of support and resistance levels
        """
        if len(df) < 30:
            return []
        
        levels = []
        
        # Get recent price data
        recent = df.iloc[-50:].copy()
        
        # Find local maxima and minima
        highs = recent['high'].values
        lows = recent['low'].values
        close = recent['close'].values
        
        current_price = close[-1]
        
        # Find resistance levels (local maxima)
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                # Only include resistance above current price
                if highs[i] > current_price * 1.005:  # 0.5% away to avoid noise
                    # Check if this level is near an existing one
                    nearby = False
                    for level in levels:
                        if level["type"] == "resistance" and abs(level["price"] - highs[i]) / level["price"] < 0.01:
                            nearby = True
                            break
                    
                    if not nearby:
                        distance_pct = (highs[i] - current_price) / current_price * 100
                        strength = self._calculate_level_strength(df, highs[i], "resistance")
                        
                        levels.append({
                            "type": "resistance",
                            "price": float(highs[i]),
                            "distance_percent": float(distance_pct),
                            "strength": strength
                        })
        
        # Find support levels (local minima)
        for i in range(2, len(lows) - 2):
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                # Only include support below current price
                if lows[i] < current_price * 0.995:  # 0.5% away to avoid noise
                    # Check if this level is near an existing one
                    nearby = False
                    for level in levels:
                        if level["type"] == "support" and abs(level["price"] - lows[i]) / level["price"] < 0.01:
                            nearby = True
                            break
                    
                    if not nearby:
                        distance_pct = (current_price - lows[i]) / current_price * 100
                        strength = self._calculate_level_strength(df, lows[i], "support")
                        
                        levels.append({
                            "type": "support",
                            "price": float(lows[i]),
                            "distance_percent": float(distance_pct),
                            "strength": strength
                        })
        
        # Sort by distance from current price
        levels.sort(key=lambda x: x["distance_percent"])
        
        # Return only closest levels
        return levels[:6]
    
    def _calculate_level_strength(self, df: pd.DataFrame, price_level: float, level_type: str) -> float:
        """
        Calculate the strength of a support or resistance level based on historical tests.
        
        Args:
            df: DataFrame with price data
            price_level: The price level to evaluate
            level_type: "support" or "resistance"
            
        Returns:
            Strength value between 0.0 and 1.0
        """
        # Price band around the level (1% range)
        band_range = price_level * 0.01
        lower_band = price_level - band_range
        upper_band = price_level + band_range
        
        tests = 0
        total_volume = 0
        
        # Count how many times price interacted with this level
        for i in range(len(df) - 1):
            row = df.iloc[i]
            
            if level_type == "support":
                # For support, check if price came close to or below level then bounced
                if row['low'] <= upper_band and row['close'] > price_level:
                    tests += 1
                    total_volume += row['volume']
            else:
                # For resistance, check if price came close to or above level then reversed
                if row['high'] >= lower_band and row['close'] < price_level:
                    tests += 1
                    total_volume += row['volume']
        
        # Calculate strength based on tests and volume
        # More tests and higher volume increase strength
        base_strength = min(tests / 5, 1.0)  # Cap at 1.0
        
        # Adjust for recency (if available in recent data, it's more relevant)
        recent_df = df.iloc[-20:]
        recent_tests = 0
        
        for i in range(len(recent_df) - 1):
            row = recent_df.iloc[i]
            
            if level_type == "support":
                if row['low'] <= upper_band and row['close'] > price_level:
                    recent_tests += 1
            else:
                if row['high'] >= lower_band and row['close'] < price_level:
                    recent_tests += 1
        
        # Recent tests have higher weight
        recent_factor = min(recent_tests / 2, 1.0)
        
        # Final strength is weighted average
        strength = (base_strength * 0.6) + (recent_factor * 0.4)
        
        return round(strength, 2)
    
    def _create_context_description(self, snapshot: Dict) -> str:
        """
        Create a text description of the market context for vectorization and retrieval.
        
        Args:
            snapshot: Market context snapshot
            
        Returns:
            Text description of market context
        """
        if not snapshot or "symbol" not in snapshot:
            return ""
        
        symbol = snapshot["symbol"]
        description_parts = [f"Market context for {symbol}"]
        
        # Add timeframe information
        for timeframe, tf_data in snapshot.get("timeframes", {}).items():
            price = tf_data.get("close")
            if price is None:
                continue
                
            description_parts.append(f"{timeframe} timeframe:")
            
            # Price information
            description_parts.append(f"Price: {price:.2f}")
            
            # Trend information
            ema12 = tf_data.get("indicators", {}).get("ema_12")
            ema26 = tf_data.get("indicators", {}).get("ema_26")
            
            if ema12 is not None and ema26 is not None:
                if ema12 > ema26:
                    description_parts.append(f"Bullish trend with EMA12 above EMA26")
                else:
                    description_parts.append(f"Bearish trend with EMA12 below EMA26")
            
            # RSI information
            rsi = tf_data.get("indicators", {}).get("rsi")
            if rsi is not None:
                if rsi > 70:
                    description_parts.append(f"Overbought RSI at {rsi:.1f}")
                elif rsi < 30:
                    description_parts.append(f"Oversold RSI at {rsi:.1f}")
                else:
                    description_parts.append(f"Neutral RSI at {rsi:.1f}")
            
            # MACD information
            macd = tf_data.get("indicators", {}).get("macd")
            macd_signal = tf_data.get("indicators", {}).get("macd_signal")
            
            if macd is not None and macd_signal is not None:
                if macd > macd_signal:
                    description_parts.append(f"Bullish MACD crossover")
                else:
                    description_parts.append(f"Bearish MACD crossover")
            
            # Bollinger Bands information
            bb_lower = tf_data.get("indicators", {}).get("bb_lower")
            bb_upper = tf_data.get("indicators", {}).get("bb_upper")
            
            if bb_lower is not None and bb_upper is not None:
                if price < bb_lower:
                    description_parts.append(f"Price below lower Bollinger Band, potentially oversold")
                elif price > bb_upper:
                    description_parts.append(f"Price above upper Bollinger Band, potentially overbought")
        
        # Add pattern information
        for pattern in snapshot.get("patterns", []):
            description_parts.append(f"Pattern detected: {pattern.get('pattern')} on {pattern.get('timeframe')} timeframe")
            description_parts.append(pattern.get("description", ""))
        
        # Add support/resistance information
        for level in snapshot.get("aggregated_signals", {}).get("support_resistance", []):
            level_type = level.get("type", "")
            price = level.get("price", 0)
            strength = level.get("strength", 0)
            
            description_parts.append(f"{level_type.capitalize()} level at {price:.2f} with strength {strength:.2f}")
        
        # Add aggregated signals
        agg_signals = snapshot.get("aggregated_signals", {})
        
        short_term = agg_signals.get("short_term_trend")
        if short_term:
            description_parts.append(f"Short-term trend: {short_term}")
        
        medium_term = agg_signals.get("medium_term_trend")
        if medium_term:
            description_parts.append(f"Medium-term trend: {medium_term}")
        
        volatility = agg_signals.get("volatility")
        if volatility:
            description_parts.append(f"Volatility: {volatility}")
        
        momentum = agg_signals.get("momentum")
        if momentum:
            description_parts.append(f"Momentum: {momentum}")
        
        # Join all parts into a single description
        return " ".join(description_parts)
    
    def store_context(self, context: Dict) -> None:
        """
        Store a market context in the database.
        
        Args:
            context: Market context snapshot
        """
        if not context or "symbol" not in context:
            logger.warning("Invalid context data, not storing")
            return
        
        # Add to context database
        self.context_db["contexts"].append(context)
        self.context_db["last_updated"] = datetime.now().isoformat()
        
        # Update metadata
        self.context_db["metadata"]["total_contexts"] = len(self.context_db["contexts"])
        
        symbol = context["symbol"]
        if symbol not in self.context_db["metadata"]["symbols"]:
            self.context_db["metadata"]["symbols"].append(symbol)
        
        # Save to disk
        self._save_context_db()
        
        # Update vectorizer with new context
        self._update_vectorizer()
    
    def _update_vectorizer(self) -> None:
        """Update the TF-IDF vectorizer with current context database."""
        contexts = self.context_db["contexts"]
        
        if not contexts:
            logger.warning("No contexts available for vectorization")
            return
        
        # Extract text descriptions
        descriptions = [context.get("text_description", "") for context in contexts]
        descriptions = [desc for desc in descriptions if desc]  # Filter out empty descriptions
        
        if not descriptions:
            logger.warning("No valid text descriptions found for vectorization")
            return
        
        try:
            # Fit or update vectorizer
            self.vectorizer.fit(descriptions)
            
            # Transform descriptions to vectors
            self.context_vectors = self.vectorizer.transform(descriptions)
            
            # Save updated vectorizer
            self._save_vectorizer()
            
            logger.info(f"Vectorizer updated with {len(descriptions)} context descriptions")
        except Exception as e:
            logger.error(f"Error updating vectorizer: {e}")
    
    def find_similar_contexts(self, 
                            query_context: Dict, 
                            top_n: int = 3, 
                            min_similarity: float = 0.3) -> List[Dict]:
        """
        Find similar historical market contexts based on the query context.
        
        Args:
            query_context: Current market context to find matches for
            top_n: Number of top matches to return
            min_similarity: Minimum similarity score to include in results
            
        Returns:
            List of similar historical contexts with similarity scores
        """
        if self.context_vectors is None or "text_description" not in query_context:
            logger.warning("Vectorizer not initialized or query missing text description")
            return []
        
        query_text = query_context["text_description"]
        
        try:
            # Transform query to vector
            query_vector = self.vectorizer.transform([query_text])
            
            # Calculate similarity with all stored contexts
            similarities = cosine_similarity(query_vector, self.context_vectors).flatten()
            
            # Find top matches
            contexts = self.context_db["contexts"]
            matches = []
            
            # Create (index, similarity) pairs and sort by similarity
            similarity_pairs = [(i, similarities[i]) for i in range(len(similarities))]
            similarity_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Select top matches above minimum similarity
            for idx, similarity in similarity_pairs[:top_n]:
                if similarity >= min_similarity:
                    match = contexts[idx].copy()
                    match["similarity_score"] = float(similarity)
                    matches.append(match)
            
            return matches
        except Exception as e:
            logger.error(f"Error finding similar contexts: {e}")
            return []
    
    def prune_old_contexts(self, max_age_days: int = None) -> None:
        """
        Prune old contexts from the database to maintain reasonable size.
        
        Args:
            max_age_days: Maximum age in days of contexts to keep
        """
        if max_age_days is None:
            max_age_days = self.config.get("rag", {}).get("context_retention_days", 30)
        
        if not self.context_db["contexts"]:
            return
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        cutoff_str = cutoff_date.isoformat()
        
        # Filter contexts
        orig_count = len(self.context_db["contexts"])
        self.context_db["contexts"] = [
            ctx for ctx in self.context_db["contexts"]
            if ctx.get("timestamp", "") > cutoff_str
        ]
        new_count = len(self.context_db["contexts"])
        
        if orig_count != new_count:
            # Update metadata
            self.context_db["metadata"]["total_contexts"] = new_count
            self.context_db["last_updated"] = datetime.now().isoformat()
            
            # Rebuild symbol list
            symbols = set()
            for ctx in self.context_db["contexts"]:
                symbols.add(ctx.get("symbol", ""))
            self.context_db["metadata"]["symbols"] = list(symbols)
            
            # Save to disk
            self._save_context_db()
            
            # Update vectorizer
            self._update_vectorizer()
            
            logger.info(f"Pruned {orig_count - new_count} old contexts, {new_count} remaining")

    def gather_market_context(self, symbol: str) -> Dict:
        """
        Gather comprehensive market context for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT")
            
        Returns:
            Dictionary containing market context information
        """
        # Collect data for multiple timeframes
        market_data = self.collect_multi_timeframe_data(symbol)
        
        if not market_data:
            logger.warning(f"No market data available for {symbol}")
            return {}
        
        # Create market snapshot
        snapshot = self.create_market_snapshot(symbol, market_data)
        
        # Store context for future retrieval
        self.store_context(snapshot)
        
        # Find similar historical contexts
        similar_contexts = self.find_similar_contexts(snapshot)
        
        # Add similar contexts to the snapshot
        snapshot["similar_contexts"] = similar_contexts
        
        return snapshot
    
    def process_trade_outcome(self, 
                             symbol: str, 
                             entry_time: str, 
                             exit_time: str,
                             entry_price: float,
                             exit_price: float,
                             direction: str,
                             profit_loss: float,
                             context_id: str = None) -> None:
        """
        Process trade outcome to update context database with success/failure information.
        
        Args:
            symbol: Trading pair symbol
            entry_time: Entry timestamp
            exit_time: Exit timestamp
            entry_price: Entry price
            exit_price: Exit price
            direction: Trade direction ('buy' or 'sell')
            profit_loss: Profit/loss amount or percentage
            context_id: Optional ID of the context used for the trade
        """
        if not self.context_db["contexts"]:
            return
        
        # Find the context that was used for the trade
        target_context = None
        
        if context_id:
            # Find by ID if provided
            for ctx in self.context_db["contexts"]:
                if ctx.get("id") == context_id:
                    target_context = ctx
                    break
        else:
            # Find closest context by timestamp
            try:
                entry_dt = datetime.fromisoformat(entry_time)
                closest_diff = timedelta(days=1)
                
                for ctx in self.context_db["contexts"]:
                    if ctx.get("symbol") != symbol:
                        continue
                    
                    ctx_time = ctx.get("timestamp", "")
                    try:
                        ctx_dt = datetime.fromisoformat(ctx_time)
                        diff = abs(ctx_dt - entry_dt)
                        
                        if diff < closest_diff:
                            closest_diff = diff
                            target_context = ctx
                    except ValueError:
                        continue
            except ValueError:
                logger.warning(f"Invalid entry time format: {entry_time}")
                return
        
        if target_context:
            # Update context with trade outcome
            if "trade_outcomes" not in target_context:
                target_context["trade_outcomes"] = []
            
            outcome = {
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "direction": direction,
                "profit_loss": profit_loss,
                "success": profit_loss > 0
            }
            
            target_context["trade_outcomes"].append(outcome)
            
            # Update success rate for this context
            outcomes = target_context["trade_outcomes"]
            successful = sum(1 for o in outcomes if o.get("success", False))
            target_context["success_rate"] = successful / len(outcomes) if outcomes else 0
            
            # Save updated database
            self._save_context_db()
            
            logger.info(f"Updated context with trade outcome: {profit_loss} {'profit' if profit_loss > 0 else 'loss'}")