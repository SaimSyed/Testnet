"""
Trading strategy module implementing various technical strategy classes,
strategy management, and position sizing.
"""

import numpy as np
import pandas as pd
from datetime import datetime

class Strategy:
    """Base strategy class that all specific strategies inherit from."""
    
    def __init__(self, weight=1.0):
        """
        Initialize the strategy.
        
        Args:
            weight (float): Weight of this strategy in a multi-strategy setup (0.0-1.0)
        """
        self.weight = weight
    
    def generate_signal(self, data):
        """
        Generate trading signal from market data.
        
        Args:
            data (pd.DataFrame): DataFrame with price and indicator data
            
        Returns:
            dict: Signal information
        """
        raise NotImplementedError("Subclasses must implement generate_signal()")
    
    def _create_signal_template(self):
        """Create a base signal template with common fields."""
        return {
            'timestamp': datetime.now().timestamp(),
            'direction': None,  # 'buy', 'sell', or None
            'strength': 0.0,    # Signal strength 0.0-1.0
            'confidence': 0.0,  # Confidence level 0.0-1.0
            'source': self.__class__.__name__,
            'signal_type': 'technical',
            'weight': self.weight
        }


class CrossoverStrategy(Strategy):
    """
    Moving average crossover strategy that generates signals when
    a faster MA crosses over a slower MA.
    """
    
    def __init__(self, fast_period=20, slow_period=50, weight=1.0):
        """
        Initialize the crossover strategy.
        
        Args:
            fast_period (int): Fast moving average period
            slow_period (int): Slow moving average period
            weight (float): Strategy weight
        """
        super().__init__(weight)
        self.fast_period = fast_period
        self.slow_period = slow_period
    
    def generate_signal(self, data):
        """
        Generate crossover signals.
        
        Args:
            data (pd.DataFrame): DataFrame with price and indicator data
            
        Returns:
            dict: Signal information
        """
        signal = self._create_signal_template()
        
        # Check if we have enough data and required columns
        required_cols = [f'sma_{self.fast_period}', f'sma_{self.slow_period}']
        if len(data) < 2 or not all(col in data.columns for col in required_cols):
            signal['confidence'] = 0.0
            return signal
        
        # Get current and previous values
        fast_current = data[f'sma_{self.fast_period}'].iloc[-1]
        slow_current = data[f'sma_{self.slow_period}'].iloc[-1]
        fast_prev = data[f'sma_{self.fast_period}'].iloc[-2]
        slow_prev = data[f'sma_{self.slow_period}'].iloc[-2]
        
        # Check for crossovers
        if fast_prev <= slow_prev and fast_current > slow_current:
            # Bullish crossover
            signal['direction'] = 'buy'
            signal['strength'] = 1.0
            signal['confidence'] = 0.7
            signal['signal_type'] = 'bullish_crossover'
        elif fast_prev >= slow_prev and fast_current < slow_current:
            # Bearish crossover
            signal['direction'] = 'sell'
            signal['strength'] = 1.0
            signal['confidence'] = 0.7
            signal['signal_type'] = 'bearish_crossover'
        else:
            # No crossover, but still provide trend information
            if fast_current > slow_current:
                # In bullish trend
                signal['direction'] = 'hold_long'
                signal['strength'] = 0.3
                signal['confidence'] = 0.5
                signal['signal_type'] = 'bullish_trend'
            else:
                # In bearish trend
                signal['direction'] = 'hold_short'
                signal['strength'] = 0.3
                signal['confidence'] = 0.5
                signal['signal_type'] = 'bearish_trend'
        
        return signal


class RsiStrategy(Strategy):
    """
    Relative Strength Index (RSI) strategy that generates signals
    based on overbought/oversold conditions.
    """
    
    def __init__(self, period=14, overbought=70, oversold=30, weight=1.0):
        """
        Initialize the RSI strategy.
        
        Args:
            period (int): RSI period
            overbought (float): RSI level considered overbought
            oversold (float): RSI level considered oversold
            weight (float): Strategy weight
        """
        super().__init__(weight)
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def generate_signal(self, data):
        """
        Generate RSI-based signals.
        
        Args:
            data (pd.DataFrame): DataFrame with price and indicator data
            
        Returns:
            dict: Signal information
        """
        signal = self._create_signal_template()
        
        # Check if we have enough data and required columns
        if len(data) < 2 or 'rsi' not in data.columns:
            signal['confidence'] = 0.0
            return signal
        
        # Get current and previous RSI values
        rsi_current = data['rsi'].iloc[-1]
        rsi_prev = data['rsi'].iloc[-2]
        
        # Check for oversold/overbought conditions
        if rsi_current < self.oversold:
            # Oversold condition
            if rsi_prev < rsi_current:
                # RSI turning up from oversold - strong buy signal
                signal['direction'] = 'buy'
                signal['strength'] = 1.0
                signal['confidence'] = 0.8
                signal['signal_type'] = 'rsi_oversold_reversal'
            else:
                # Still oversold but not turning up yet
                signal['direction'] = 'buy'
                signal['strength'] = 0.7
                signal['confidence'] = 0.6
                signal['signal_type'] = 'rsi_oversold'
        elif rsi_current > self.overbought:
            # Overbought condition
            if rsi_prev > rsi_current:
                # RSI turning down from overbought - strong sell signal
                signal['direction'] = 'sell'
                signal['strength'] = 1.0
                signal['confidence'] = 0.8
                signal['signal_type'] = 'rsi_overbought_reversal'
            else:
                # Still overbought but not turning down yet
                signal['direction'] = 'sell'
                signal['strength'] = 0.7
                signal['confidence'] = 0.6
                signal['signal_type'] = 'rsi_overbought'
        else:
            # RSI in neutral territory (not oversold or overbought)
            # Check for trend direction
            if rsi_current > 50:
                # Bullish bias
                signal['direction'] = 'hold_long'
                signal['strength'] = 0.3
                signal['confidence'] = 0.4
                signal['signal_type'] = 'rsi_bullish_bias'
            else:
                # Bearish bias
                signal['direction'] = 'hold_short'
                signal['strength'] = 0.3
                signal['confidence'] = 0.4
                signal['signal_type'] = 'rsi_bearish_bias'
        
        return signal


class StrategyManager:
    """
    Manages multiple strategies and combines their signals
    with weighted aggregation.
    """
    
    def __init__(self):
        """Initialize the strategy manager with an empty strategy list."""
        self.strategies = []
    
    def add_strategy(self, strategy):
        """
        Add a strategy to the manager.
        
        Args:
            strategy (Strategy): Strategy instance to add
        """
        self.strategies.append(strategy)
    
    def generate_signals(self, data):
        """
        Generate signals from all strategies and combine them.
        
        Args:
            data (pd.DataFrame): DataFrame with price and indicator data
            
        Returns:
            list: Combined signals from all strategies
        """
        if not self.strategies:
            return []
        
        # Generate individual signals
        signals = []
        for strategy in self.strategies:
            signals.append(strategy.generate_signal(data))
        
        return signals
    
    def get_composite_signal(self, signals):
        """
        Calculate a composite signal from multiple strategy signals.
        
        Args:
            signals (list): List of signal dictionaries
            
        Returns:
            float: Composite signal strength (-1.0 to 1.0)
        """
        if not signals:
            return 0.0
        
        # Normalize weights
        total_weight = sum(signal.get('weight', 1.0) for signal in signals)
        if total_weight == 0:
            return 0.0
        
        # Calculate weighted signals
        composite = 0.0
        for signal in signals:
            direction_multiplier = 0
            if signal.get('direction') in ['buy', 'hold_long']:
                direction_multiplier = 1
            elif signal.get('direction') in ['sell', 'hold_short']:
                direction_multiplier = -1
            
            weight = signal.get('weight', 1.0) / total_weight
            strength = signal.get('strength', 0.0)
            confidence = signal.get('confidence', 0.5)
            
            # Add weighted contribution to composite signal
            composite += direction_multiplier * strength * confidence * weight
        
        return composite


class PositionSizer:
    """
    Position sizing utility implementing various risk-based approaches
    to calculate appropriate position sizes.
    """
    
    def __init__(self, account_balance=10000, risk_per_trade=0.02, 
                 max_position_size=0.2, method='fixed_risk'):
        """
        Initialize the position sizer.
        
        Args:
            account_balance (float): Available trading capital
            risk_per_trade (float): Maximum risk per trade as fraction of account (e.g., 0.02 = 2%)
            max_position_size (float): Maximum position size as fraction of account
            method (str): Position sizing method ('fixed_risk', 'fixed_percent', 'kelly')
        """
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.method = method
    
    def calculate_position_size(self, signal, current_price, atr=None):
        """
        Calculate position size based on the selected method.
        
        Args:
            signal (dict): Signal information
            current_price (float): Current asset price
            atr (float): Average True Range for volatility-based sizing
            
        Returns:
            dict: Position sizing information
        """
        if self.method == 'fixed_risk':
            return self._fixed_risk_position(signal, current_price, atr)
        elif self.method == 'fixed_percent':
            return self._fixed_percent_position(signal, current_price)
        elif self.method == 'kelly':
            return self._kelly_position(signal, current_price)
        else:
            # Default to fixed percent if method is not recognized
            return self._fixed_percent_position(signal, current_price)
    
    def _fixed_risk_position(self, signal, current_price, atr=None):
        """
        Calculate position size where risk is fixed at a percentage of account.
        Uses ATR to determine stop-loss distance if available.
        
        Args:
            signal (dict): Signal information
            current_price (float): Current asset price
            atr (float): Average True Range for volatility-based stop placement
            
        Returns:
            dict: Position sizing information
        """
        risk_amount = self.account_balance * self.risk_per_trade
        
        # Determine stop-loss distance
        stop_distance = 0
        if atr is not None and atr > 0:
            # Use ATR for stop distance, typically 2-3 times ATR
            atr_multiplier = 2.5
            stop_distance = atr * atr_multiplier
        else:
            # Default to fixed percentage of price
            stop_distance = current_price * 0.05  # 5% stop
        
        # Calculate position size based on risk per pip
        if stop_distance > 0:
            position_value = risk_amount / (stop_distance / current_price)
        else:
            position_value = 0
        
        # Cap at maximum position size
        max_position_value = self.account_balance * self.max_position_size
        if position_value > max_position_value:
            position_value = max_position_value
        
        # Calculate units based on current price
        if current_price > 0:
            units = position_value / current_price
        else:
            units = 0
        
        # Build result
        result = {
            'position_value': position_value,
            'units': units,
            'risk_amount': risk_amount,
            'stop_price': current_price - stop_distance if signal.get('direction') == 'buy' else current_price + stop_distance,
            'stop_distance': stop_distance,
            'percent_of_account': position_value / self.account_balance if self.account_balance > 0 else 0
        }
        
        return result
    
    def _fixed_percent_position(self, signal, current_price):
        """
        Calculate position size as a fixed percentage of account.
        Adjusts percentage based on signal confidence.
        
        Args:
            signal (dict): Signal information
            current_price (float): Current asset price
            
        Returns:
            dict: Position sizing information
        """
        # Base position percentage on signal confidence
        confidence = signal.get('confidence', 0.5)
        base_percent = self.max_position_size * confidence
        
        # Calculate position value
        position_value = self.account_balance * base_percent
        
        # Calculate units based on current price
        if current_price > 0:
            units = position_value / current_price
        else:
            units = 0
        
        # Build result
        result = {
            'position_value': position_value,
            'units': units,
            'risk_amount': position_value * 0.05,  # Assuming 5% risk from entry
            'percent_of_account': base_percent
        }
        
        return result
    
    def _kelly_position(self, signal, current_price):
        """
        Calculate position size using the Kelly Criterion.
        Requires win rate and risk/reward ratio.
        
        Args:
            signal (dict): Signal information
            current_price (float): Current asset price
            
        Returns:
            dict: Position sizing information
        """
        # For Kelly, we need win rate and risk/reward ratio
        # If these aren't available in the signal, use defaults
        win_rate = signal.get('historical_success_rate', 0.5)
        risk_reward = signal.get('risk_reward_ratio', 1.5)
        
        # Basic Kelly formula: f = (bp - q) / b
        # Where f is fraction to bet, b is odds received (risk/reward), 
        # p is probability of win, and q is probability of loss (1-p)
        
        if risk_reward > 0:
            kelly_fraction = (win_rate * risk_reward - (1 - win_rate)) / risk_reward
        else:
            kelly_fraction = 0
        
        # Kelly can suggest betting nothing (if negative) or too much (if very positive)
        # Constrain it to sensible limits
        kelly_fraction = max(0, kelly_fraction)  # No negative bets
        kelly_fraction = min(kelly_fraction, self.max_position_size)  # Cap at maximum
        
        # Usually, it's recommended to bet a fraction of the Kelly bet (e.g., half-Kelly)
        # to reduce volatility
        position_fraction = kelly_fraction * 0.5  # Half-Kelly
        
        # Calculate position value
        position_value = self.account_balance * position_fraction
        
        # Calculate units based on current price
        if current_price > 0:
            units = position_value / current_price
        else:
            units = 0
        
        # Build result
        result = {
            'position_value': position_value,
            'units': units,
            'risk_amount': position_value * 0.05,  # Assuming 5% risk from entry
            'percent_of_account': position_fraction,
            'kelly_fraction': kelly_fraction,
            'win_rate_used': win_rate,
            'risk_reward_used': risk_reward
        }
        
        return result