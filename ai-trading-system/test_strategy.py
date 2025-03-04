from strategy.trading_strategy import CrossoverStrategy, RsiStrategy, StrategyManager
import pandas as pd
import numpy as np

# Create sample test data
dates = pd.date_range(start="2023-01-01", periods=50, freq='D')
data = pd.DataFrame({
    'open': np.random.normal(100, 5, 50),
    'high': np.random.normal(102, 5, 50),
    'low': np.random.normal(98, 5, 50),
    'close': np.random.normal(101, 5, 50),
    'volume': np.random.normal(1000, 200, 50),
    'sma_20': np.random.normal(101, 3, 50),
    'sma_50': np.random.normal(100, 2, 50),
    'rsi_14': np.random.normal(50, 15, 50),
}, index=dates)

# Create strategies
sma_strategy = CrossoverStrategy("SMA_Crossover", "sma_20", "sma_50")
rsi_strategy = RsiStrategy("RSI_Strategy", "rsi_14")

# Test individual strategies
print("Testing SMA Crossover Strategy:")
sma_result = sma_strategy.generate_signals(data)
print(f"Generated {sum(sma_result['signal'] != 0)} signals")

print("\nTesting RSI Strategy:")
rsi_result = rsi_strategy.generate_signals(data)
print(f"Generated {sum(rsi_result['signal'] != 0)} signals")

# Test strategy manager
print("\nTesting Strategy Manager:")
manager = StrategyManager()
manager.add_strategy(sma_strategy, weight=0.6)
manager.add_strategy(rsi_strategy, weight=0.4)
combined = manager.generate_combined_signals(data)
print(f"Generated {sum(combined['signal'] != 0)} combined signals")
